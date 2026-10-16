"""Configuration of the bimanual tactile environments (pydantic, frozen, as DexTaG's EnvArgs).

`BimanualEnvArgs` holds what both methods share (data, simulation, hands, objects, simulated glove);
`DexTaGEnvArgs` and `ReconEnvArgs` add each method's action, observation, reward, termination and
training-aid settings. `RlReconEnvArgs` configures rl_recon_env.py (Benchmark-Genesis' rl_recon on a per-episode
reference.npz, one or two hands) on its own. Every field can be overridden from the command line with --env.<field>=<value>.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


def frozen_config() -> ConfigDict:
    return ConfigDict(extra="forbid", frozen=True, protected_namespaces=(), validate_default=True)


class DataArgs(BaseModel):
    model_config = frozen_config()
    clip_dir: str = "data/humantouch/clips"
    clips: list[str] = []                    # clip names (index.json); empty = every clip in clip_dir
    max_clips: int = 0                       # 0 = no limit
    # Deterministic train/eval split by episode (DexTaG's split, keyed on the episode index)
    split_mode: Literal["train", "eval", "all"] = "all"
    train_ratio: float = 0.9
    split_seed: int = 0

    @field_validator("clips", mode="before")
    @classmethod
    def split_clips(cls, v):  # --env.data.clips=a,b is stored as the string (model_copy does not validate)
        return [c for c in v.split(",") if c] if isinstance(v, str) else v


class HandArgs(BaseModel):
    """Two Wuji hands, each on a 6-DoF PD-controlled virtual base (Benchmark-Genesis' floating hand)."""
    model_config = frozen_config()
    wuji_dir: str | None = None              # wuji-retargeting checkout (URDF); None: $WUJI_RETARGETING_DIR
    kp_base_pos: float = 3000.0
    kv_base_pos: float = 80.0
    kp_base_rot: float = 300.0
    kv_base_rot: float = 20.0
    kp_finger: float = 20.0
    kv_finger: float = 1.0
    finger_effort_scale: float = 2.0         # x URDF effort limits
    base_force_limit: float = 200.0
    base_torque_limit: float = 20.0
    friction: float = 1.0
    # domain randomisation of the PD gains (training only)
    kp_range: tuple[float, float] = (1.0, 1.0)
    kd_range: tuple[float, float] = (1.0, 1.0)


class ObjectArgs(BaseModel):
    model_config = frozen_config()
    friction: float = 1.0
    # The plate under each object's rest pose, object-sized: the reference hand often goes lower than
    # the object's support elsewhere in the clip (into the basket). Its top is the object's bottom at the
    # grasp onset, lowered (by at most support_max_drop) under the reference hand before the grasp.
    support_size: tuple[float, float, float] = (0.16, 0.16, 0.02)
    support_max_drop: float = 0.03
    grasp_blend_frames: int = 6              # rest pose -> in-palm pose at the grasp onset (frames)
    start_clearance: float = 0.01            # episodes start where the hands clear the resting objects (m)
    # domain randomisation (training only, DexTaG's ObjectConfig)
    mass_randomization_range: float = 0.0    # +/- fraction of the clip's mass
    friction_range: tuple[float, float] = (1.0, 1.0)
    init_pos_noise_xy: float = 0.0           # m, at reset
    init_rot_noise_deg: float = 0.0
    # random force / torque kicks while the object is lifted (DexTaG; 0 = off)
    perturbation_force_scale: float = 0.0
    perturbation_torque_scale: float = 0.0
    perturbation_prob_range: tuple[float, float] = (0.001, 0.05)
    perturbation_lift_threshold: float = 0.05
    perturbation_curriculum: tuple[int, int] = (2000, 10000)


class TactileArgs(BaseModel):
    model_config = frozen_config()
    mapping: Literal["layout", "anatomical", "mano"] = "layout"
    assets_dir: str = "assets/tactile"
    probe_radius: float = 0.004              # contact if the taxel is closer than this to the surface
    kn: float = 250.0                        # simulated pressure (glove obs units) per metre of penetration:
                                             # a taxel 4 mm into the surface reads 1, like ~30 glove units
    soft_tau: float = 0.0015                 # soft contact temperature (Benchmark-Genesis)
    glove_scale: float = 30.0                # cleaned glove units per observation unit


class BimanualEnvArgs(BaseModel):
    model_config = frozen_config()
    env_name: str
    method: Literal["dextag", "recon"]
    seed: int = 3
    data: DataArgs = DataArgs()
    hands: HandArgs = HandArgs()
    objects: ObjectArgs = ObjectArgs()
    tactile: TactileArgs = TactileArgs()
    substeps: int = 4
    work_dir: str = "outputs/urdf"           # floating-base URDFs are written here
    reward_args: dict[str, dict[str, float]]
    reward_scale_by_dt: bool = True          # DexTaG multiplies every reward scale by dt
    # episode clips (scripts/data/build_episode_clip.py: a whole demonstration with its reconstructed objects):
    # training episodes start anywhere and last this long; evaluation plays the whole demonstration
    episode_window_s: float = 6.0
    episode_manip_weight: float = 4.0        # a start inside a manipulation is this many times as likely
    # hands simulated: both, one side, or "auto" (the hands that manipulate an object in the clips); a hand
    # left out is not in the scene at all (no body, no action, no observation, no reward)
    active_hands: Literal["both", "left", "right", "auto"] = "both"


class DexTaGEnvArgs(BimanualEnvArgs):
    """DexTaG: tactile-as-guidance teacher (relative base / absolute hand actions, tactile map obs)."""
    method: Literal["dextag"] = "dextag"
    base_action_scale_pos: float = 0.02      # m per step
    base_action_scale_rot: float = 0.1       # rad per step
    action_ema: float = 0.2                  # hand target blend towards the command (1 = none)
    action_latency_range: tuple[int, int] = (0, 1)
    obs_history_len: int = 8
    obs_future_length: int = 16
    obs_future_stride: int = 1
    obs_scales: dict[str, float]
    actor_obs_terms: list[str]
    critic_obs_terms: list[str]
    proprio_terms: list[str]
    target_motion_terms: list[str]
    tactile_map_term: str = "tactile_map"
    # termination
    fingertip_failure_dist: tuple[float, float, float, float, float] = (0.057, 0.064, 0.071, 0.086, 0.086)
    object_pos_failure: float = 0.0583       # m (DexTaG: 0.02 / 0.343)
    object_rot_failure_deg: float = 43.7     # (DexTaG: 15 / 0.343)
    support_contact_thresholds: tuple[float, float, float] = (500.0, 100.0, 50.0)   # at 0 / 10k / 20k iters
    success_lookahead: int = 3
    # training aid: PD force pulling the object along its reference, decaying to 0 (DexTaG's object guide)
    object_guide_decay_steps: int = 5000
    object_guide_kp: float = 50.0            # 1/s^2 (force = mass * (kp * error - kd * velocity))
    object_guide_kd: float = 10.0


class ReconEnvArgs(BimanualEnvArgs):
    """Benchmark-Genesis rl_recon: residual policy on the retargeted reference, contact-F1 reward."""
    method: Literal["recon"] = "recon"
    reward_scale_by_dt: bool = False
    action_scale_pos: float = 0.03
    action_scale_rot: float = 0.25
    action_scale_finger: float = 0.4
    obs_future: tuple[int, ...] = (1, 2, 4)
    obs_tactile_future: tuple[int, ...] = (1, 5)
    glove_obs: bool = True                   # False: GT glove contacts zeroed in the observation (ablation)
    # episodes
    start_frame_prob: float = 0.5            # random start before the first grasp (else frame 0)
    rsi_grasp_prob: float = 0.5              # start inside a grasp with the object in the reference hand
    term_wrist: float = 0.10
    term_kp: float = 0.06
    term_obj: float = 0.20
    obj_depth_discount: float = 3.0          # the object error along the head-camera ray counts 1/x
    # training aid: hand-anchored virtual object controller, decayed linearly to 0 over the first
    # `assist_decay` fraction of the iterations (0 = off); evaluation never uses it
    assist_init: float = 1.0
    assist_decay: float = 0.5
    assist_tau: float = 0.1
    assist_gain: float = 0.5


class ReferenceDataArgs(BaseModel):
    model_config = frozen_config()
    reference: str = ""                        # reference.npz (scripts/rl/prepare_reference.py)


class RlReconEnvArgs(BaseModel):
    """Benchmark-Genesis' rl_recon (reconstruction/rl_recon/env.py: EnvConfig) on a per-episode reference
    (tv2s_data/reference.py), one or two hands (rl_recon_env.py). Defaults are rl_recon's."""
    model_config = frozen_config()
    env_name: str = "RlReconEnv"
    method: Literal["recon"] = "recon"
    seed: int = 3
    data: ReferenceDataArgs = ReferenceDataArgs()
    hands: HandArgs = HandArgs()
    tactile: TactileArgs = TactileArgs(mapping="mano")   # probe radius, soft contact; the taxels are the reference's
    substeps: int = 4
    work_dir: str = "outputs/urdf"
    # hands simulated: every hand of the reference (the ones the demonstration uses), or one side of it
    active_hands: Literal["reference", "left", "right"] = "reference"
    # objects: their collision meshes (the reference's), convex-decomposed at decompose_threshold, and their voxel SDF
    decompose_threshold: float = 0.05
    object_friction: float = 1.0
    object_mass: float = 0.0                   # kg; 0: the reference's (Benchmark-Genesis': none, so rl_recon's 80 g)
    sdf_voxel: float = 0.001
    tactile_objects: Literal["manipulated", "all"] = "manipulated"   # what the glove can touch: the hand's object
                                               # (rl_recon: its one object) or every object
    # action: residual on the reference at t+1
    action_scale_pos: float = 0.03
    action_scale_rot: float = 0.25
    action_scale_finger: float = 0.4
    # tactile reward
    tac_reach_weight: float = 0.5              # r_tactile = (1 - w) soft F1 + w reach shaping
    tac_reach_scale: float = 0.02
    tactile_reward: bool = True                # False: w_tac counts as 0 (ablation, train.py --no-tactile; still logged)
    glove_obs: bool = True                     # False: the GT glove contacts ahead are zeroed (train.py --no-glove-obs)
    # reward weights / scales
    w_tac: float = 1.5
    w_hand: float = 1.0
    w_obj: float = 0.5
    w_obj_axis: float = 0.75
    w_rate: float = 0.02
    sigma_wrist_pos: float = 0.02
    sigma_wrist_rot: float = 0.25
    sigma_kp: float = 0.03
    obj_margin_across: float = 0.02
    obj_scale_across: float = 0.05
    obj_margin_along: float = 0.10
    obj_scale_along: float = 0.10
    obj_axis_margin: float = 0.26              # rad
    obj_axis_scale: float = 0.5
    obj_rot_weight: float = 0.0                # hinge on the full rotation error (off, as rl_recon)
    obj_rot_margin: float = 0.6
    obj_depth_discount: float = 3.0            # the object error along the camera ray counts 1/x (termination)
    # episodes
    start_frame_prob: float = 0.5              # random start where the objects rest (else frame 0)
    rest_starts: Literal["before_first_grasp", "away_from_holds"] = "before_first_grasp"   # rl_recon: the first
    rest_margin_s: float = 0.33                # ... at least this far from a grasp (rl_recon: 10 frames)
    rsi_grasp_prob: float = 0.5                # start at a grasp frame, the object in the reference hand
    rsi_min_contact_frac: float = 0.5          # ... where the glove fires >= this x the hand's peak
    rsi_min_clearance: float = 0.003           # ... and the held object clears the table (m)
    start_clearance: float = -1.0              # starts need the reference taxels this far from resting objects (m);
                                               # < 0: the reference's (this pipeline's: 5 mm; Benchmark-Genesis': off)
    episode_window_s: float = 0.0              # training episodes last this long; 0: to the clip end (rl_recon)
    eval_start: Literal["first", "in_hand"] = "first"   # evaluation: from frame 0, or in hand (rl_recon's two)
    term_wrist: float = 0.10
    term_kp: float = 0.06
    term_obj: float = 0.20
    # training aid: hand-anchored virtual object controller, decayed linearly to 0 over the first
    # `assist_decay` fraction of the iterations; evaluation never uses it
    assist_init: float = 1.0
    assist_decay: float = 0.5
    assist_tau: float = 0.1
    assist_gain: float = 0.5
    obs_future: tuple[int, ...] = (1, 2, 4)
    obs_tactile_future: tuple[int, ...] = (1, 5)


ENV_ARGS_CLASSES = {"DexTaGBimanualEnv": DexTaGEnvArgs, "ReconBimanualEnv": ReconEnvArgs, "RlReconEnv": RlReconEnvArgs}


def load_env_args(path) -> BaseModel:
    """A saved env configuration (configs/env_args.json), parsed with its environment's class."""
    import json
    from pathlib import Path

    text = Path(path).read_text()
    return ENV_ARGS_CLASSES[json.loads(text)["env_name"]].model_validate_json(text)
