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
scripts/data/         extract_humantouch.py, retarget_humantouch.py, cut_clips.py (stage 1),
                      build_sim_clips.py (stage 2)
scripts/reconstruction/ semantic_parsing.py (Gemini scene parsing of a video)
scripts/tactile/      build_tactile_mappings.py, compare_tactile_mappings.py
scripts/policy/       run_ppo_dextag.py, run_ppo_recon.py, summarize_runs.py, visualize_policy.py, common.py
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

1. extract the episodes, converting HumanTouch's Unity world (left-handed, y up) to a right-handed z-up
   frame (read as is, every skeleton had the chirality of the other hand and gravity was sideways);
2. retarget both skeletons onto the Wuji hands at 30 Hz (fingertips 12-18 mm median from the human);
3. **cut** the episodes into clips (`cut_clips.py` -> `cuts/`): manipulation cycles from the Gemini scene
   file when the episode has one, else from the semantic annotations; per clip the glove without its
   drifting offset, each hand's grasp, and a video of the head camera (no hand pose needed);
4. build the **simulation clips** (`build_sim_clips.py` -> `clips/`): per cut the retargeted reference
   hands and a **proxy object** per grasping hand (HumanTouch has no object pose: see docs/METHODS.md);
5. place the 360 assigned glove cells of each hand on the Wuji with both mappings
   (`assets/tactile/{layout,anatomical}_{left,right}.npz` and a check figure per hand).

### The clip files, stage by stage

```
video ──semantic_parsing.py──> scene/<key>.json            objects (SAM3 prompts, material, size), clip hints
video + glove (+ scene) ──cut_clips.py──> cuts/<name>.{json,npz} + cuts/videos/<name>.mp4
                                          window, caption, objects per hand, glove at every video frame
cuts + retargeted hands ──build_sim_clips.py──> clips/<name>.pkl    (HumanTouch: what the RL envs load)
```

```bash
export GEMINI_API_KEY=...
~/miniconda3/envs/sam3d/bin/python scripts/reconstruction/semantic_parsing.py \
    --video ~/HumanTouch/X001/videos/chunk-000/observation.images.cam_head/episode_000002.mp4 \
    --out data/humantouch/scene/episode_000002.json
$PY_SIM scripts/data/cut_clips.py --dataset humantouch --data data/humantouch --root ~/HumanTouch/X001 --episodes 2
$PY_SIM scripts/data/build_sim_clips.py --data data/humantouch --episodes 2
```

`semantic_parsing.py` needs only the video. Gemini watches it twice and writes one JSON: `objects`
(every object with an id, a SAM3 text prompt `description`, a time `visible_at_s` where it is
unoccluded, material, shape, size) and `clips` (one per object transfer: start / end hints, caption, per
hand the object manipulated, its source and destination, the objects visible).
`cut_clips.py` cuts an episode from these hints when its scene file exists (else from the dataset's
segmentation; `--segments scene|dataset` forces one) and keeps, per cut, Gemini's description and the
glove at every video frame (HumanTouch: cleaned pressure of the 360 cells and the 460 raw cells;
TouchScale: force and shear of the 880 taxels, in N), plus the head-camera video with the caption, the
object Gemini names per hand and the glove grasp timeline. `build_sim_clips.py` adds the retargeted hands
and the proxy objects; its `index.json` flags per hand whether the glove grasp and Gemini agree
(`glove_vs_gemini`: both / glove_only / gemini_only / none). Both stages only replace the episodes they
process (`--clean` to start over).

### Objects from the video: SAM3 masks, depth, RecGen meshes

```bash
launch/reconstruct_objects.sh humantouch 2 data/humantouch       # after semantic_parsing.py on episode 2
launch/reconstruct_objects.sh touchscale 5a1ae97d                # TouchScale: its sensor depth, no MoGe
```

From the episode's scene file (`scene/<key>.json`), in `perception/<key>/`:
1. `prepare_frames.py`: the video sampled at 10 fps, 720 px high, undistorted (HumanTouch's calibration),
   `intrinsics.yaml`; TouchScale's registered sensor depth for the same frames;
2. `segment_objects_sam3.py` (`sam3` env): one video mask per scene object, prompted with its Gemini
   description on the frame nearest `visible_at_s` and propagated both ways (`sam3/<id>.npz`,
   `sam3/overlay.mp4`);
3. `estimate_depth_moge.py` (`moge` env): MoGe metric depth with the camera's real field of view;
4. `reconstruct_objects_recgen.py` (`recgen` env): a RecGen mesh per movable object from its best frame
   (largest unclipped mask with depth near `visible_at_s`): `recgen/<id>/posed_mesh.obj` in the camera frame
   of that frame; `objects.json` compares each mesh's extent with Gemini's size estimate.

### TouchScale: scene parsing, cuts and videos

```bash
export GEMINI_API_KEY=...
launch/touchscale_clips.sh 5a1ae97d,03af5dc0        # episode ids (or prefixes); default: MAX_EPISODES=5
SKIP_PARSE=1 launch/touchscale_clips.sh             # no Gemini: one cut per episode, its task segment
EXTRACT=1 launch/touchscale_clips.sh                # first extract the downloaded shard .tar files
```

TouchScale (2077AIDataFoundation/TouchScale, `$TOUCHSCALE_ROOT`) records a head RGB-D camera, two wrist
cameras and two tactile gloves (15 pads per hand, in N) per short task, but **no hand pose**: its
episodes go through the same Gemini parsing (with the dataset's task and items as context) and the same
stage 1 (`cut_clips.py --dataset touchscale` -> `data/touchscale/cuts/`; contact: >= 5 taxels above 0.2 N,
which reproduces the dataset's contact fraction), but not stage 2: simulating them needs a video
hand-pose estimator first.

### EgoTouch: the whole pipeline, 213 tasks

```bash
export GEMINI_API_KEY=...
python scripts/data/download_egotouch.py --list                         # tasks and episode counts
DOWNLOAD=1 TASKS=pick_up_bottle,twist_bottle_cap PER_TASK=5 launch/prepare_egotouch.sh
launch/reconstruct_objects.sh egotouch pick_up_bottle__20260314_000533_921
$PY_SIM scripts/data/build_episode_clip.py --perception data/egotouch/perception/pick_up_bottle__20260314_000533_921 \
    --hands mocap --data data/egotouch --out data/episodes/et_bottle
launch/train_recon.sh et_bottle --env.data.clip_dir=data/episodes/et_bottle --env.data.clips=20260314000533921_episode
```

EgoTouch (zhouzhoujy/EgoTouch, `$EGOTOUCH_ROOT`, `src/tv2s_data/egotouch.py`) records short bimanual
tasks at 30 Hz: a head camera (320x240 or 640x480, no intrinsics), two wrist cameras, Rokoko gloves
(21 joints per hand), Vive trackers (chest, wrists) and two pressure gloves (a 21x21 hand-shaped grid:
151 taxels, 15 bend sensors). What differs from HumanTouch:
- **hands**: the Rokoko skeletons (Unity world, converted like HumanTouch's) are the mocap; Vive lives in
  another world and its wrist trackers are often frozen, so they are not used for the hands;
- **camera**: placed by `register_camera.py --dataset egotouch` as W @ chest tracker @ X with W, X and the
  focal length fitted on the dataset's glove masks (from many random starts; a static camera when there
  is no chest track, or when it fits better); the focal is weakly constrained by masks (a prior,
  `--focal-prior`);
- **glove**: `assets/tactile/egotouch_glove.json` (cell list: region, u, v per taxel, read by
  `load_glove_layout`); contact above 0.1 of the normalised grid; mapping `mano_egotouch` (glove -> MANO ->
  Wuji, `build_tactile_mappings.py --dataset egotouch`), picked by the environment from the clip;
- **quality**: `extract_egotouch.py` reports each stream's valid share and skips episodes whose Rokoko
  stream stalls (e.g. pick_up_mouse); some episodes have no glove masks (SAM3 segments the hands then).

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

### rl_recon on one demonstration (per-episode reference)

`launch/rl_recon.sh NAME VIDEO` reproduces Benchmark-Genesis' RL reconstruction (`reconstruction/rl_recon`):

1. Gemini says which hands the demonstration uses (`hands_used` in the scene file).
2. `scripts/rl/prepare_reference.py` writes the episode's `reference.npz` (format: `src/tv2s_data/reference.py`) for those hands only. One hand used gives a one-hand environment.
3. PPO trains on the env `rl_recon` (`rl_recon_env.py`). It follows `env.py` term by term: on a Benchmark-Genesis reference, the observation has the same 681 values.
4. `scripts/policy/eval_rl_recon.py` scores every checkpoint as `train.py` does (from frame 0 and in hand, no assist, no termination, over 32 envs) and keeps `model_best.pt` / `model_best_inhand.pt`.

The reference has three sources:
- **`EGOTOUCH=<key>`** (recommended for EgoTouch, where it trained best): the episode is reconstructed by Benchmark-Genesis' own chain, `launch/reconstruct_bg.sh` (its `run_touchanything_pipeline.sh`, unchanged: camera motion from CoTracker on the table, the mocap placed on the episode's WiLoR 2D joints and the depth, the depth rescaled into the mocap's metres, RecGen without mask erosion, CoTracker + Kalman object trajectory), on the object the scene file says the hands manipulate; then as `BG_RECON`. Needs the Benchmark-Genesis checkout (`BG_RECON_ROOT`) and its conda envs.
- **`BG_RECON=<dir>`**: an existing Benchmark-Genesis reconstruction of the video, through its own reference chain (retargeting, `prepare_reference.py`, `mask_axes.py`, `tactile_grasp_pose.py`, `select_grasp_pose.py`, the last one keeping the in-hand pose the hand holds under physics). It handles one hand only.
- **`CLIP=<pkl>`**: an episode clip of this pipeline. It can have both hands and several objects.

```bash
export GEMINI_API_KEY=...                  # or VLM=qwen
EGOTOUCH=pick_up_screwdriver__20260321_103227_213 launch/rl_recon.sh et_screwdriver_bg   # Gemini, reconstruction, reference, PPO, eval
BG=~/Benchmark-Genesis/reconstruction/human_videos/touchanything/pick_up_screwdriver_103227
BG_RECON=$BG launch/rl_recon.sh screwdriver_103227 $BG/pick_up_screwdriver_rgb.mp4      # ITERATIONS=2000 N_ENVS=2048
# a fine-tune of a checkpoint (train.py --resume --reset-iterations), e.g. rl_recon's v18 stage:
SKIP_HANDS=1 SKIP_REFERENCE=1 REFERENCE=data/references/screwdriver_103227/reference.npz RESUME=logs/screwdriver_103227/<time>/checkpoints/model_best_inhand.pt FINETUNE=1 ITERATIONS=800 \
    TRAIN_ARGS="--env.assist_init=0.5 --env.assist_decay=0.3" launch/rl_recon.sh screwdriver_103227_ft $BG/pick_up_screwdriver_rgb.mp4
```

## 3. Compare

```bash
launch/compare_tactile_mappings.sh       # policy-free: geometry, replay agreement, reward sensitivity
launch/compare_methods.sh study          # method x mapping: 4 trainings, one summary table
```

## 4. Watch the policies

```bash
launch/visualize_policies.sh ep000002_dextag,ep000002_recon       # one video per clip, runs stacked
launch/visualize_policies.sh ep000002_recon --clips=ep000002_clip03 --viewer   # interactive viewer
```

On top, the human demonstration (HumanTouch's head-camera video, frame-synchronised with the clip;
only `observation.images.cam_head` is needed). Each run is replayed as in its evaluation
(deterministic, aids off). Per clip: an egocentric camera at
the recorded head and a front camera, the reference hand keypoints and object axis as green spheres,
and per hand the human glove pressure and its agreement with the simulated glove (green: both, blue:
glove only, red: simulation only). Videos and trajectories (npz) go to `outputs/videos/<name>/`.
`scripts/policy/render_rollouts.py` (used by the Jean Zay jobs) is the lighter alternative: one 3D
view with the reference hands as ghosts.

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
