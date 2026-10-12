# Tactile-Video2Sim

Bimanual, tactile-guided reinforcement learning on the **HumanTouch** egocentric dataset (glove tactile
maps + hand skeletons), with the two RL methods of

- **DexTaG** (`~/DexTaG`, *Tactile-as-Guidance in RL for Dexterous Manipulation*): a teacher policy trained
  with PPO to retarget glove demonstrations, rewarded for matching the glove's **tactile map**;
- **Benchmark-Genesis rl_recon** (`~/Benchmark-Genesis/reconstruction/rl_recon`): a **residual** PPO
  policy on the retargeted human hand that reconstructs a hand-object trajectory, rewarded with a **soft
  contact F1** against the glove;

and the two ways these repos map a human glove onto a robot hand, compared on the same data.
The code follows DexTaG's structure: pydantic configs in registries, environments whose observations
and rewards are selected by name, reward-term classes that declare the state they read, a PPO agent
with a runner, and command lines with dot-notation overrides.

Both methods run on the same embodiment: **two Wuji hands** (left and right), each on a 6-DoF
PD-controlled virtual base, in Genesis. See [docs/METHODS.md](docs/METHODS.md) for what was carried over
from each repo, what had to change for HumanTouch, and the mapping comparison.

## Layout

```
src/tv2s_data/        HumanTouch I/O, glove layout + signal cleaning, clips and proxy objects
src/tv2s_env/
  robots/             Wuji kinematics (URDF FK, collision hulls, floating-base URDF)
  tactile/            glove -> Wuji mappings (layout: DexTaG, anatomical: Benchmark-Genesis), simulated glove
  common/rewards/     dextag_terms.py (DexTaG), recon_terms.py (rl_recon), as RewardTerm classes
  sim/robots/         FloatingWujiHand (Genesis entity on a virtual base)
  sim/envs/           BimanualTactileEnv (shared scene/state/metrics), DexTaGBimanualEnv, ReconBimanualEnv,
                      config/{schema,registry}.py
src/tv2s_agent/       PPO (DexTaG's, with rl_recon's options), temporal encoder / MLP, GAE buffer, runner
scripts/data/         extract_humantouch.py, retarget_humantouch.py, build_clips.py
scripts/tactile/      build_tactile_mappings.py, compare_tactile_mappings.py
scripts/policy/       run_ppo_dextag.py, run_ppo_recon.py, summarize_runs.py, common.py
launch/               the shell entry points below (config.sh holds paths and conda envs)
configs/retargeting/  Wuji retargeting configs (Benchmark-Genesis' wuji_mano_{left,right}.yaml)
assets/tactile/       generated glove -> Wuji mappings (both hands, both methods)
```

## Environments

Three existing conda envs are used (paths in `launch/config.sh`, all overridable):

| env | used for |
|---|---|
| `lerobot` | reading the HumanTouch parquet files (pandas, pyarrow) |
| `wuji_retargeting` | Wuji retargeting (pinocchio, `~/wuji-ego-mint/.../wuji-retargeting`) |
| `genesis` | clips, mappings, simulation and training (genesis-world 1.3.3, torch, trimesh) |

Stock Genesis is enough: DexTaG's forked `TactileField` sensor and `TactileVisualizer` are replaced by a
torch glove sensor (`src/tv2s_env/tactile/sensor.py`).

## 1. Prepare HumanTouch

```bash
launch/prepare_humantouch.sh                 # first 20 episodes (MAX_EPISODES=20)
launch/prepare_humantouch.sh 1-50            # or an explicit list / ranges
```

1. extract the episodes, mirroring the **left-handed** source world to a right-handed one (every
   skeleton otherwise has the chirality of the other hand: the retargeted Wuji fingers bent backwards);
2. retarget both skeletons onto the Wuji hands at 30 Hz (fingertips 3-23 mm median from the human);
3. cut the episodes into manipulation cycles (semantic annotations), remove the glove's drifting
   offset, detect each hand's grasp, and infer a **proxy object** per grasping hand (HumanTouch has no
   object pose: see docs/METHODS.md);
4. place the 360 assigned glove cells of each hand on the Wuji with both mappings
   (`assets/tactile/{layout,anatomical}_{left,right}.npz` and a check figure per hand).

## 2. Train

```bash
# DexTaG teacher (temporal encoder, tactile-map reward, object guide force, ...)
launch/train_dextag.sh dextag_v1                       # NUM_ENVS=1024 ITERATIONS=15000
# Benchmark-Genesis rl_recon (residual policy, contact-F1 reward, assist, RSI)
launch/train_recon.sh recon_v1                         # NUM_ENVS=2048 ITERATIONS=1500
```

Each script trains, then evaluates (one environment per clip, deterministic, training aids off). Useful
knobs: `SPLIT=train` (train on the train split, evaluate on the held-out episodes), `ENV_NAME=...` (the
`*_anatomical` / `*_layout` variants swap the glove mapping), rl_recon ablations `TACTILE=0`,
`GLOVE_OBS=0`, `ASSIST=0`. Any config field can be overridden on the command line:

```bash
launch/train_dextag.sh my_run --env.tactile.mapping=anatomical --algo.lr=3e-4 \
    --reward_args.WeightedGaussianBlurredTactileMapSimilarityReward.scale=0 --env.data.max_clips=20
```

Runs go to `logs/<exp>/<timestamp>/` (`configs/`, `checkpoints/`, `progress.csv`, `tb/` for TensorBoard,
`eval/`). The Python entry points are `scripts/policy/run_ppo_{dextag,recon}.py` (`--eval`,
`--all_checkpoints`, `--split_mode=eval`, `--resume_checkpoint=latest`).

## 3. Compare

```bash
launch/compare_tactile_mappings.sh       # policy-free: geometry, replay agreement, reward sensitivity
launch/compare_methods.sh study          # method x mapping: 4 trainings, one summary table
```

## Jean Zay: the study with wandb and rollout videos

```bash
launch/prepare_humantouch.sh 1-20                       # data in $SCRATCH/HumanTouch/X001 (launch/config.sh)
nohup launch/wandb_sync_loop.sh > logs/wandb_sync.log 2>&1 &   # login node: uploads the offline runs
launch/submit_study.sh smoke s0                         # 4 short jobs on the dev QoS: checks the pipeline
launch/submit_study.sh full study1                      # DexTaG/rl_recon x layout/anatomical, one H100 each
```

Each job (`launch/slurm/train_eval.slurm`) trains with wandb (`TV2S_WANDB=1`, offline: compute nodes have
no internet), renders rollout videos of fixed clips from the checkpoints as they appear
(`scripts/policy/render_rollouts.py`: 3D view with the reference hands as orange ghosts, simulated vs
real glove images), resumes after the 20 h limit, then evaluates and logs the evaluation
(`scripts/policy/wandb_report.py eval`). A last job logs the method x mapping comparison run
(`wandb_report.py compare`); `wandb_report.py mapping outputs/tactile_comparison` logs the policy-free
mapping comparison. Runs of one study share a wandb group (the prefix).

## Limits worth knowing

- The objects are proxies (cylinders inferred from the glove and the retargeted hand), not the real
  objects; their sizes saturate at the 4.5 cm radius bound on wide power grasps.
- The glove layout gives region labels only; which end of a finger patch is the fingertip is an
  assumption (`--finger-tip-row`, default 22 = the row farthest from the palm).
- No student/distillation stage (DexTaG's BC student needs a wrist camera and an arm): the DexTaG side
  is the teacher (retargeter).
