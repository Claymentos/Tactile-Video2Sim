"""Helpers shared by the training scripts (DexTaG's scripts/policy/utils.py, plus the evaluation both
methods are scored with).

Command line: `--key=value` pairs. Known script arguments are passed to main(); the others are config
overrides in dot notation, routed by prefix: --env.<field>, --reward_args.<Term>.<arg>, --algo.<field>,
--runner.<field> (DexTaG's apply_overrides_generic).
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
# paths in the configs (clips, tactile assets, logs) are relative to the repository root
os.chdir(REPO)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_cli(argv: list[str]) -> dict[str, str]:
    """['--a=1', '--b.c', '2', '--flag'] -> {'a': '1', 'b.c': '2', 'flag': 'True'}."""
    out, i = {}, 0
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith("--"):
            raise SystemExit(f"unexpected argument {arg}")
        key = arg[2:]
        if "=" in key:
            key, val = key.split("=", 1)
        elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            val = argv[i + 1]
            i += 1
        else:
            val = "True"
        out[key.replace("-", "_") if "." not in key else key] = val
        i += 1
    return out


def split_overrides(cli: dict[str, str], main_args: set[str]) -> tuple[dict, dict, dict, dict]:
    """Script arguments, env overrides, algo overrides, runner overrides."""
    script, env, algo, runner = {}, {}, {}, {}
    for k, v in cli.items():
        if k.startswith(("env.", "reward_args.")):
            env[k] = v
        elif k.startswith("algo."):
            algo[k] = v
        elif k.startswith("runner."):
            runner[k] = v
        elif k in main_args:
            script[k] = parse_scalar(v)
        else:
            raise SystemExit(f"unknown argument --{k} (script args: {sorted(main_args)})")
    return script, env, algo, runner


def parse_scalar(value: Any) -> Any:
    """Best-effort parse of CLI string into bool/int/float, else return as-is."""
    if isinstance(value, int | float | bool):
        return value
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in {"true", "false"}:
            return v.lower() == "true"
        try:
            if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                return int(v)
            return float(v)
        except ValueError:
            return value
    return value


def strip_prefixes(path: str, prefixes: tuple[str, ...]) -> str:
    # repeatedly strip any matching prefix until none match
    changed = True
    while changed:
        changed = False
        for pref in prefixes:
            if path.startswith(pref):
                path = path[len(pref) :]
                changed = True
    return path


def tokenize_path(path: str, prefixes: tuple[str, ...]) -> list[str]:
    """Split a dot/bracket path into tokens after removing known prefixes."""
    path = strip_prefixes(path, prefixes)
    out: list[str] = []
    buf = ""
    i = 0
    while i < len(path):
        c = path[i]
        if c == ".":
            if buf:
                out.append(buf)
                buf = ""
            i += 1
            continue
        if c == "[":
            if buf:
                out.append(buf)
                buf = ""
            j = path.find("]", i)
            if j == -1:
                buf += c
                i += 1
            else:
                out.append(path[i + 1 : j])
                i = j + 1
            continue
        buf += c
        i += 1
    if buf:
        out.append(buf)
    return out


def parse_sequence_like(raw: Any, target_example: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    parts = [p.strip() for p in raw.split(",")]
    parsed = [parse_scalar(p) for p in parts]
    if isinstance(target_example, tuple):
        return tuple(parsed)
    return parsed


def deep_apply(obj: Any, tokens: list[str], raw_val: Any) -> Any:
    """Deeply apply a single override value addressed by tokens to a Pydantic model.

    Handles nested BaseModel, dicts (up to two levels, e.g. reward_args.<Term>.scale),
    and lists/tuples (by index).
    """
    from pydantic import BaseModel as _BM  # type: ignore

    if not tokens:
        return obj

    field = tokens[0]
    rest = tokens[1:]

    # getattr guard
    try:
        current = getattr(obj, field)
    except AttributeError:
        return obj

    # Leaf assignment
    if not rest:
        new_value = parse_scalar(raw_val)
        if isinstance(current, tuple | list) and isinstance(raw_val, str) and "," in raw_val:
            new_value = parse_sequence_like(raw_val, current)
        if isinstance(current, tuple) and isinstance(new_value, list):
            new_value = tuple(new_value)
        return obj.model_copy(update={field: new_value}) if isinstance(obj, _BM) else obj

    # Dict field (e.g., reward_args)
    if isinstance(current, dict):
        key = rest[0]
        new_dict = dict(current)
        if len(rest) == 1:
            new_dict[key] = parse_scalar(raw_val)
        elif isinstance(new_dict.get(key), dict):
            sub = dict(new_dict[key])
            sub_key = rest[1]
            if len(rest) == 2:
                sub[sub_key] = parse_scalar(raw_val)
            else:
                # deeper-than-two-level dict nesting still falls back to a joined key
                sub[".".join(rest[1:])] = parse_scalar(raw_val)
            new_dict[key] = sub
        else:
            new_dict[".".join(rest)] = parse_scalar(raw_val)
        return obj.model_copy(update={field: new_dict})

    # List/Tuple field
    if isinstance(current, list | tuple):
        try:
            idx = int(rest[0])
        except ValueError:
            return obj
        if idx < 0 or idx >= len(current):
            return obj
        seq_list = list(current)
        elem = seq_list[idx]
        if len(rest) == 1:
            new_elem = parse_scalar(raw_val)
        else:
            if hasattr(elem, "model_copy"):
                new_elem = deep_apply(elem, rest[1:], raw_val)
            elif isinstance(elem, dict):
                new_elem = dict(elem)
                new_elem[rest[1]] = parse_scalar(raw_val)
            else:
                new_elem = elem
        seq_list[idx] = new_elem
        new_seq = tuple(seq_list) if isinstance(current, tuple) else seq_list
        return obj.model_copy(update={field: new_seq})

    # Nested Pydantic model
    if hasattr(current, "model_copy"):
        updated_sub = deep_apply(current, rest, raw_val)
        return obj.model_copy(update={field: updated_sub})

    return obj


def apply_overrides_generic(
    base_obj: Any, overrides: dict[str, Any] | None, prefixes: tuple[str, ...]
) -> Any:
    if not overrides:
        return base_obj
    updated = base_obj
    for raw_key, raw_val in overrides.items():
        tokens = tokenize_path(raw_key, prefixes)
        updated = deep_apply(updated, tokens, raw_val)
    return updated



def save_config(config: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=1))


def load_config(cls: Any, path: Path) -> Any:
    return cls.model_validate_json(path.read_text())


def latest_run_dir(exp_dir: Path) -> Path:
    """Most recent run under logs/<exp_name>/ that has a checkpoint."""
    runs = sorted([d for d in exp_dir.glob("*") if list(d.glob("checkpoints/*.pt"))], key=lambda d: d.name)
    if not runs:
        raise FileNotFoundError(f"no run with checkpoints under {exp_dir}")
    return runs[-1]


def create_env(env_args: Any, num_envs: int, device: str = "cuda", eval_mode: bool = False,
               show_viewer: bool = False, render: dict | None = None,
               camera_res: tuple[int, int] | None = None) -> Any:
    import genesis as gs
    if not gs._initialized:  # noqa: SLF001
        gs.init(backend=gs.gpu if device == "cuda" else gs.cpu, precision="32", logging_level="warning",
                seed=env_args.seed, performance_mode=True)
    import tv2s_env.sim.envs as envs
    return getattr(envs, env_args.env_name)(env_args, num_envs=num_envs, show_viewer=show_viewer,
                                            device=torch.device(device), eval_mode=eval_mode, render=render,
                                            camera_res=camera_res)


@torch.no_grad()
def evaluate_policy(env: Any, policy: Any, out_path: Path | None = None, place_tol: float = 0.06) -> dict[str, Any]:
    """One deterministic episode per environment (the eval env has one env per clip), aids off.

    The same metrics for both methods: success (the episode reached the clip end with every object within
    `place_tol` of its reference there: in the hand, or put down on its placing support), completion (fraction
    of the clip played before termination), contact precision / recall / F1 of the simulated glove
    against the real one, the glove-image similarity (DexTaG's reward without its scale), and the palm,
    fingertip and object errors against the reference."""
    B = env.num_envs
    env.reset()
    obs, _ = env.get_observations()
    running = torch.ones(B, dtype=torch.bool, device=env.device)
    keys = ["tac_tp", "tac_sim", "tac_real", "tac_map_similarity", "e_wrist", "e_tip", "e_kp", "e_obj", "obj_valid"]
    sums = {k: torch.zeros(B, device=env.device) for k in keys + ["reward", "obj_frames", "map_frames"]}
    steps = torch.zeros(B, device=env.device)
    success = torch.zeros(B, dtype=torch.bool, device=env.device)
    start = env.start.clone()
    length = (env.horizon - env.start).float().clamp(min=1)
    while running.any():
        action, _ = policy(obs, deterministic=True)
        obs, _, reward, terminated, truncated, info = env.step(action)
        m, r = info["metrics"], running.float()
        for k in keys:
            if k == "e_obj":
                sums[k] += m[k] * m["obj_valid"] * r
            elif k == "tac_map_similarity":
                sums[k] += m[k] * (m["tac_real"] > 0) * r
            else:
                sums[k] += m[k] * r
        sums["obj_frames"] += m["obj_valid"] * r
        sums["map_frames"] += (m["tac_real"] > 0).float() * r
        sums["reward"] += reward[:, 0] * r
        steps += r
        done = (terminated | truncated)[:, 0] & running
        reached_end = truncated[:, 0] | info["termination"].get("succeeded", torch.zeros_like(done))
        success |= done & reached_end & (m["e_obj_end"] < place_tol)
        running &= ~done
    per_clip = []
    for i in range(B):
        tp, ns, nr = (sums[k][i].item() for k in ("tac_tp", "tac_sim", "tac_real"))
        p, rc = tp / max(ns, 1), tp / max(nr, 1)
        per_clip.append(dict(
            clip=env.clip_names[int(env.env_clip[i])], start=int(start[i]), steps=int(steps[i]),
            success=bool(success[i]), completion=min(1.0, steps[i].item() / length[i].item()),
            reward=sums["reward"][i].item() / max(steps[i].item(), 1),
            tac_precision=p, tac_recall=rc, tac_f1=2 * p * rc / max(p + rc, 1e-9),
            tac_map_similarity=sums["tac_map_similarity"][i].item() / max(sums["map_frames"][i].item(), 1),
            e_wrist_cm=100 * sums["e_wrist"][i].item() / max(steps[i].item(), 1),
            e_tip_mm=1000 * sums["e_tip"][i].item() / max(steps[i].item(), 1),
            e_obj_cm=100 * sums["e_obj"][i].item() / max(sums["obj_frames"][i].item(), 1),
        ))
    summary = {k: float(np.mean([c[k] for c in per_clip])) for k in per_clip[0] if k not in ("clip", "start", "steps")}
    result = dict(summary=summary, clips=per_clip)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=1))
    return result


def print_eval(result: dict[str, Any], header: str = "") -> None:
    s = result["summary"]
    print(f"\n{'=' * 100}\n{header}")
    print(f"{'clip':<22} {'succ':>5} {'compl':>6} {'F1':>5} {'P':>5} {'R':>5} {'map':>5} {'wrist cm':>8} "
          f"{'tip mm':>7} {'obj cm':>7}")
    for c in result["clips"]:
        print(f"{c['clip']:<22} {int(c['success']):>5} {c['completion']:>6.2f} {c['tac_f1']:>5.2f} "
              f"{c['tac_precision']:>5.2f} {c['tac_recall']:>5.2f} {c['tac_map_similarity']:>5.2f} "
              f"{c['e_wrist_cm']:>8.1f} {c['e_tip_mm']:>7.0f} {c['e_obj_cm']:>7.1f}")
    print(f"{'MEAN':<22} {s['success']:>5.2f} {s['completion']:>6.2f} {s['tac_f1']:>5.2f} {s['tac_precision']:>5.2f} "
          f"{s['tac_recall']:>5.2f} {s['tac_map_similarity']:>5.2f} {s['e_wrist_cm']:>8.1f} {s['e_tip_mm']:>7.0f} "
          f"{s['e_obj_cm']:>7.1f}\n{'=' * 100}")


# ====================================================================== train / evaluate
def build_configs(method: str, env_name: str, env_ov: dict, algo_ov: dict, runner_ov: dict) -> tuple[Any, Any, Any]:
    from tv2s_agent.algos.config.registry import AlgoArgsRegistry
    from tv2s_agent.runners.config.registry import RunnerArgsRegistry
    from tv2s_env.sim.envs.config.registry import EnvArgsRegistry

    env_args = EnvArgsRegistry[env_name]
    if env_args.method != method:
        raise SystemExit(f"env {env_name} belongs to method {env_args.method}, not {method}")
    env_args = apply_overrides_generic(env_args, env_ov, prefixes=("env.",))
    algo = apply_overrides_generic(AlgoArgsRegistry[method], algo_ov, prefixes=("algo.",))
    runner = apply_overrides_generic(RunnerArgsRegistry[method], runner_ov, prefixes=("runner.",))
    return env_args, algo, runner


def fit_networks(algo: Any, env: Any) -> Any:
    """Fill the temporal encoders' input splits from the env's observation layout (DexTaG)."""
    from tv2s_agent.modules.config.schema import TemporalEncoderConfig

    upd = {}
    for role, key in (("actor", "policy_backbone"), ("critic", "critic_backbone")):
        cfg = getattr(algo, key)
        if isinstance(cfg, TemporalEncoderConfig):
            upd[key] = cfg.with_layout(env.obs_layout(role))
    return algo.model_copy(update=upd)


def train_policy(method: str, exp_name: str, env_name: str, num_envs: int, cli_overrides: tuple,
                 resume_checkpoint: str | None = None, device: str = "cuda", reset_iterations: bool = False,
                 lightning: bool = False) -> None:
    """`reset_iterations` (with `resume_checkpoint`): a fine-tune, as rl_recon's train.py --resume --reset-iterations:
    the network only (a fresh optimiser), iterations counted from 0 (the assist schedule restarts).
    `lightning`: the same PPO trained by a Lightning Trainer (tv2s_agent.runners.lightning_runner) instead of
    OnPolicyRunner; same checkpoints, progress.csv and wandb curves."""
    from datetime import datetime

    from tv2s_agent.algos.ppo import PPO
    from tv2s_agent.runners.onpolicy_runner import MetricLogger, OnPolicyRunner

    env_args, algo, runner_args = build_configs(method, env_name, *cli_overrides)
    seed_everything(env_args.seed)
    env = create_env(env_args, num_envs, device)
    algo = fit_networks(algo, env)
    ppo = PPO(env, algo, torch.device(device))
    start = 0
    if resume_checkpoint:
        ckpt = Path(resume_checkpoint) if resume_checkpoint != "latest" else \
            sorted((latest_run_dir(REPO / "logs" / exp_name) / "checkpoints").glob("*.pt"))[-1]
        ppo.load(ckpt, load_optimizer=not reset_iterations)
        if reset_iterations:
            ppo.current_iter = 0
        start = ppo.current_iter
        print(f"Resumed from {ckpt} at iteration {start}" + (" (fine-tune: iterations reset)" if reset_iterations else ""))
    run_dir = REPO / "logs" / exp_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_config(env_args, run_dir / "configs" / "env_args.json")
    save_config(algo, run_dir / "configs" / "algo_cfg.json")
    save_config(runner_args, run_dir / "configs" / "runner_args.json")
    print(f"Run directory: {run_dir}")
    mapping = env_args.tactile.mapping
    wandb_init = dict(name=exp_name, group=os.environ.get("WANDB_RUN_GROUP") or None,
                      job_type="train", tags=[method, mapping, env_name],
                      config=dict(method=method, mapping=mapping, env_name=env_name, num_envs=num_envs,
                                  env=json.loads(env_args.model_dump_json()), algo=json.loads(algo.model_dump_json()),
                                  runner=json.loads(runner_args.model_dump_json()),
                                  trainer="lightning" if lightning else "onpolicy"))
    if lightning:
        from tv2s_agent.runners.lightning_runner import train_lightning
        summary = train_lightning(ppo, runner_args, run_dir, wandb_init, start_iteration=start)
    else:
        runner = OnPolicyRunner(ppo, runner_args, run_dir)
        logger = MetricLogger(run_dir, wandb_init)
        try:
            summary = runner.train(logger, start_iteration=start)
        finally:
            logger.finish()
    (run_dir / "TRAINING_DONE").touch()
    print(f"Training done: {summary['iterations']} iterations in {summary['total_time'] / 60:.1f} min")


def count_clips(env_args: Any) -> int:
    """Number of clips the env will load (the eval env has one env per clip)."""
    from tv2s_env.sim.envs.manipulation.bimanual_tactile_env import select_clip_names
    return len(select_clip_names(env_args.data))


def evaluate_run(method: str, exp_name: str, checkpoint: str | None = None, all_checkpoints: bool = False,
                 split_mode: str | None = None, env_overrides: dict | None = None, device: str = "cuda") -> None:
    """Evaluate a trained run on every selected clip (one env per clip). `all_checkpoints` scores every
    checkpoint of the run and copies the best one (by contact F1, then completion) to model_best.pt, as
    Benchmark-Genesis' recon_rsl_eval.py --all."""
    import shutil

    from tv2s_agent.algos.config.schema import PPOArgs
    from tv2s_agent.algos.ppo import PPO
    from tv2s_env.sim.envs.config.schema import load_env_args

    run_dir = latest_run_dir(REPO / "logs" / exp_name)
    env_args = load_env_args(run_dir / "configs" / "env_args.json")
    if split_mode:
        env_args = env_args.model_copy(update={"data": env_args.data.model_copy(update={"split_mode": split_mode})})
    env_args = apply_overrides_generic(env_args, env_overrides or {}, prefixes=("env.",))
    algo = load_config(PPOArgs, run_dir / "configs" / "algo_cfg.json")
    ckpts = sorted((run_dir / "checkpoints").glob("checkpoint_*.pt"))
    if checkpoint:
        ckpts = [Path(checkpoint)]
    elif not all_checkpoints:
        ckpts = ckpts[-1:]
    env = create_env(env_args, num_envs=count_clips(env_args), device=device, eval_mode=True)
    ppo = PPO(env, algo, torch.device(device))
    best, best_key = None, None
    for ck in ckpts:
        ppo.load(ck, load_optimizer=False)
        res = evaluate_policy(env, ppo.get_inference_policy(), run_dir / "eval" / f"{ck.stem}.json")
        print_eval(res, f"{exp_name} / {ck.name} ({len(res['clips'])} clips, split {env_args.data.split_mode})")
        key = (round(res["summary"]["tac_f1"], 3), res["summary"]["completion"])
        if best_key is None or key > best_key:
            best, best_key = ck, key
    if all_checkpoints and best is not None:
        shutil.copy(best, run_dir / "checkpoints" / "model_best.pt")
        print(f"best: {best.name} (F1 {best_key[0]:.3f}, completion {best_key[1]:.2f}) -> model_best.pt")
