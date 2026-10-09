"""Proximal Policy Optimization (DexTaG's gs_agent/algos/ppo.py).

Separate actor / critic networks and optimizers, GAE, clipped surrogate, analytic Gaussian KL driving a
KL-adaptive learning rate once per iteration. The options of Benchmark-Genesis' rl_recon/ppo.py are
switches in PPOArgs: running observation normalisation with a std floor (updated after each update,
since the stored log-probs were computed under the old one), value clipping, a near-zero actor output
layer, and its KL thresholds.
"""

import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tv2s_agent.algos.config.schema import LearningRateType, PPOArgs
from tv2s_agent.buffers.gae_buffer import GAEBuffer
from tv2s_agent.buffers.schema import GAEBufferKey
from tv2s_agent.modules.models import RunningNorm, create_network
from tv2s_agent.modules.policies import GaussianPolicy, StateValueFunction


class PPO:
    def __init__(self, env: Any, cfg: PPOArgs, device: torch.device) -> None:
        self.env, self.cfg, self.device = env, cfg, device
        self._num_envs, self._num_steps = env.num_envs, cfg.rollout_length
        actor_obs, critic_obs = env.get_observations()
        self._actor_obs_dim, self._critic_obs_dim = actor_obs.shape[-1], critic_obs.shape[-1]
        self._action_dim = env.action_dim
        self.current_iter = 0
        self._current_lr = cfg.lr
        self._rewbuffer: deque = deque(maxlen=100)
        self._lenbuffer: deque = deque(maxlen=100)
        self._curr_reward_sum = torch.zeros(self._num_envs, device=device)
        self._curr_ep_len = torch.zeros(self._num_envs, device=device)
        self._build_actor_critic()
        self._rollouts = GAEBuffer(num_envs=self._num_envs, max_steps=self._num_steps,
                                   actor_obs_size=self._actor_obs_dim, critic_obs_size=self._critic_obs_dim,
                                   action_size=self._action_dim, device=device,
                                   gae_gamma=cfg.gamma, gae_lam=cfg.gae_lambda)

    def _build_actor_critic(self) -> None:
        cfg = self.cfg
        norm = lambda d: RunningNorm(d, std_floor=cfg.obs_norm_std_floor) if cfg.obs_normalization else None
        self._actor_norm, self._critic_norm = norm(self._actor_obs_dim), norm(self._critic_obs_dim)
        policy_backbone = create_network(cfg.policy_backbone, self._actor_obs_dim)
        self._actor = GaussianPolicy(policy_backbone, self._action_dim, cfg.initial_log_std,
                                     cfg.actor_output_init, self._actor_norm).to(self.device)
        critic_backbone = create_network(cfg.critic_backbone, self._critic_obs_dim, output_dim=1)
        self._critic = StateValueFunction(critic_backbone, self._critic_norm).to(self.device)
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=self._current_lr)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=cfg.value_lr or cfg.lr)
        n = lambda m: sum(p.numel() for p in m.parameters())
        print(f"Actor: {n(self._actor) / 1e6:.2f} M params, critic: {n(self._critic) / 1e6:.2f} M params")

    # ------------------------------------------------------------------ rollouts
    def _collect_rollouts(self) -> dict[str, Any]:
        actor_obs, critic_obs = self.env.get_observations()
        reward_terms, metrics, terminations = [], [], []
        with torch.inference_mode():
            for _ in range(self._num_steps):
                action, log_prob, mu, sigma = self._actor.forward_with_dist_params(actor_obs)
                value = self._critic(critic_obs)
                next_actor_obs, next_critic_obs, reward, terminated, truncated, infos = self.env.step(action)
                # time-outs are not failures: bootstrap them with the value of the state before the step
                reward_gae = reward + self.cfg.gamma * value * truncated.float()
                self._rollouts.append({
                    GAEBufferKey.ACTOR_OBS: actor_obs, GAEBufferKey.CRITIC_OBS: critic_obs,
                    GAEBufferKey.ACTIONS: action, GAEBufferKey.REWARDS: reward_gae,
                    GAEBufferKey.DONES: (terminated | truncated), GAEBufferKey.VALUES: value,
                    GAEBufferKey.ACTION_LOGPROBS: log_prob, GAEBufferKey.MU: mu, GAEBufferKey.SIGMA: sigma,
                })
                self._curr_reward_sum += reward[:, 0]
                self._curr_ep_len += 1
                done = (terminated | truncated)[:, 0]
                if done.any():
                    self._rewbuffer.extend(self._curr_reward_sum[done].tolist())
                    self._lenbuffer.extend(self._curr_ep_len[done].tolist())
                    self._curr_reward_sum[done] = 0
                    self._curr_ep_len[done] = 0
                reward_terms.append({k: v.mean().item() for k, v in infos["reward_terms"].items()})
                metrics.append({k: v.mean().item() for k, v in infos["metrics"].items()})
                terminations.append({k: v.float().mean().item() for k, v in infos["termination"].items()})
                actor_obs, critic_obs = next_actor_obs, next_critic_obs
            self._rollouts.set_final_value(self._critic(critic_obs))
        mean = lambda rows: {k: statistics.mean(r[k] for r in rows) for k in rows[0]}
        m = mean(metrics)
        p, r = m["tac_tp"] / max(m["tac_sim"], 1e-9), m["tac_tp"] / max(m["tac_real"], 1e-9)
        m["tac_f1"] = 2 * p * r / max(p + r, 1e-9)
        return dict(mean_reward=statistics.mean(self._rewbuffer) if self._rewbuffer else 0.0,
                    mean_ep_len=statistics.mean(self._lenbuffer) if self._lenbuffer else 0.0,
                    reward_terms=mean(reward_terms), metrics=m, termination=mean(terminations))

    # ------------------------------------------------------------------ update
    def _train_one_batch(self, mb: dict[GAEBufferKey, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg
        actor_obs, critic_obs = mb[GAEBufferKey.ACTOR_OBS], mb[GAEBufferKey.CRITIC_OBS]
        adv, ret = mb[GAEBufferKey.ADVANTAGES], mb[GAEBufferKey.RETURNS]
        dist = self._actor.dist_from_obs(actor_obs)
        new_log_prob = dist.log_prob(mb[GAEBufferKey.ACTIONS]).sum(-1, keepdim=True)
        ratio = torch.exp(new_log_prob - mb[GAEBufferKey.ACTION_LOGPROBS])
        policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio)).mean()
        old_mu, old_sigma = mb[GAEBufferKey.MU], mb[GAEBufferKey.SIGMA]
        new_mu, new_sigma = dist.mean, dist.stddev
        kl = (torch.log(new_sigma / old_sigma) + (old_sigma ** 2 + (old_mu - new_mu) ** 2) / (2 * new_sigma ** 2)
              - 0.5).sum(-1).mean()
        values = self._critic(critic_obs)
        if cfg.value_clip:
            old_v = mb[GAEBufferKey.VALUES]
            v_clip = old_v + (values - old_v).clamp(-cfg.clip_ratio, cfg.clip_ratio)
            value_loss = torch.max((values - ret) ** 2, (v_clip - ret) ** 2).mean()
        else:
            value_loss = (ret - values).pow(2).mean()
        entropy = dist.entropy().sum(-1).mean()
        loss = policy_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy
        self._actor_optimizer.zero_grad()
        self._critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._actor.parameters(), cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self._critic.parameters(), cfg.max_grad_norm)
        self._critic_optimizer.step()
        if self.current_iter >= cfg.critic_warmup_iters:
            self._actor_optimizer.step()
        return dict(policy_loss=policy_loss.item(), value_loss=value_loss.item(), entropy=entropy.item(),
                    kl_mean=kl.item())

    def _update_learning_rate(self, kl_mean: float) -> None:
        cfg = self.cfg
        if cfg.lr_type != LearningRateType.ADAPTIVE:
            return
        if kl_mean > cfg.target_kl * cfg.kl_high:
            self._current_lr = max(self._current_lr / cfg.lr_adaptive_factor, cfg.lr_min)
        elif kl_mean < cfg.target_kl * cfg.kl_low:
            self._current_lr = min(self._current_lr * cfg.lr_adaptive_factor, cfg.lr_max)
        for g in self._actor_optimizer.param_groups:
            g["lr"] = self._current_lr

    def train_one_iteration(self) -> dict[str, Any]:
        t0 = time.time()
        rollout = self._collect_rollouts()
        t1 = time.time()
        stats = [self._train_one_batch(mb) for mb in self._rollouts.minibatch_gen(self.cfg.num_mini_batches,
                                                                                  self.cfg.num_epochs)]
        if self._actor_norm is not None:
            self._actor_norm.update(self._rollouts._buffer[GAEBufferKey.ACTOR_OBS].flatten(0, 1))
            self._critic_norm.update(self._rollouts._buffer[GAEBufferKey.CRITIC_OBS].flatten(0, 1))
        self._rollouts.reset()
        mean = lambda k: statistics.mean(s[k] for s in stats)
        self._update_learning_rate(mean("kl_mean"))
        self.current_iter += 1
        steps = self._num_steps * self._num_envs
        return {
            "rollout": dict(mean_reward=rollout["mean_reward"], mean_length=rollout["mean_ep_len"]),
            "train": dict(policy_loss=mean("policy_loss"), value_loss=mean("value_loss"), entropy=mean("entropy"),
                          kl_mean=mean("kl_mean"), learning_rate=self._current_lr,
                          mean_std=self._actor.action_std.mean().item()),
            "speed": dict(rollout_fps=steps / max(t1 - t0, 1e-9), rollout_time=t1 - t0, train_time=time.time() - t1,
                          rollout_step=steps),
            "metrics": rollout["metrics"], "reward_terms": rollout["reward_terms"], "termination": rollout["termination"],
        }

    # ------------------------------------------------------------------ checkpoints
    def save(self, path: Path, extra: dict | None = None) -> None:
        torch.save(dict(model_state_dict=self._actor.state_dict(), critic_state_dict=self._critic.state_dict(),
                        actor_optimizer_state_dict=self._actor_optimizer.state_dict(),
                        critic_optimizer_state_dict=self._critic_optimizer.state_dict(),
                        iter=self.current_iter, current_lr=self._current_lr, **(extra or {})), path)

    def load(self, path: Path, load_optimizer: bool = True) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self._actor.load_state_dict(ck["model_state_dict"])
        self._critic.load_state_dict(ck["critic_state_dict"])
        if load_optimizer:
            self._actor_optimizer.load_state_dict(ck["actor_optimizer_state_dict"])
            self._critic_optimizer.load_state_dict(ck["critic_optimizer_state_dict"])
        self.current_iter = ck["iter"]
        self._current_lr = ck.get("current_lr", self._current_lr)

    def get_inference_policy(self) -> GaussianPolicy:
        self._actor.eval()
        return self._actor
