"""Helpers shared by the training and evaluation scripts (DexTaG's scripts/policy/utils.py).

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
# paths in the configs (references, tactile assets, logs) are relative to the repository root
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


# ====================================================================== train
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
    from tv2s_agent.runners.rollout_videos import RolloutVideos

    env_args, algo, runner_args = build_configs(method, env_name, *cli_overrides)
    seed_everything(env_args.seed)
    env = create_env(env_args, num_envs, device)
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
    videos = RolloutVideos(REPO / "scripts" / "policy" / "visualize_episode.py", exp_name, run_dir,
                           runner_args.video_interval, start=env_args.eval_start) if runner_args.video_interval > 0 else None
    mapping = env_args.tactile.mapping
    # $WANDB_RUN_NAME: the name shown in wandb (e.g. egotouch_pick_up_hammer_franka), the experiment's otherwise;
    # logs/<exp>/ (and its wandb_id) stay keyed by the experiment
    wandb_init = dict(name=os.environ.get("WANDB_RUN_NAME") or exp_name, group=os.environ.get("WANDB_RUN_GROUP") or None,
                      job_type="train", tags=[method, mapping, env_name],
                      config=dict(method=method, mapping=mapping, env_name=env_name, num_envs=num_envs,
                                  env=json.loads(env_args.model_dump_json()), algo=json.loads(algo.model_dump_json()),
                                  runner=json.loads(runner_args.model_dump_json()),
                                  trainer="lightning" if lightning else "onpolicy"))
    if lightning:
        from tv2s_agent.runners.lightning_runner import train_lightning
        summary = train_lightning(ppo, runner_args, run_dir, wandb_init, start_iteration=start, videos=videos)
    else:
        runner = OnPolicyRunner(ppo, runner_args, run_dir)
        logger = MetricLogger(run_dir, wandb_init)
        try:
            summary = runner.train(logger, start_iteration=start, videos=videos)
        finally:
            logger.finish()
    (run_dir / "TRAINING_DONE").touch()
    print(f"Training done: {summary['iterations']} iterations in {summary['total_time'] / 60:.1f} min")
