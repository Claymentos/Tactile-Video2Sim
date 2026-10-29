"""Configuration of the rl_recon environment (rl_recon_env.py: Benchmark-Genesis' rl_recon on a per-episode
reference.npz), pydantic and frozen. Every field can be overridden from the command line with --env.<field>=<value>.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


def frozen_config() -> ConfigDict:
    return ConfigDict(extra="forbid", frozen=True, protected_namespaces=(), validate_default=True)


class HandArgs(BaseModel):
    """Each Wuji hand on a 6-DoF PD-controlled virtual base (Benchmark-Genesis' floating hand)."""
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


class ArmArgs(BaseModel):
    """Each Wuji hand on a Franka Panda (franka_wuji.py) instead of the virtual base: off by default."""
    model_config = frozen_config()
    enabled: bool = False
    urdf: str | None = None                  # Panda without gripper; None: Genesis' urdf/panda_bullet/panda_nohand.urdf
    # mount: the palm frame on the flange (z along the forearm, as the palm's fingers), `mount_offset` m out of it
    mount_offset: float = 0.02
    mount_yaw: float | None = None           # rad about the flange axis; None: the best multiple of 30 deg by IK
    # base on the table (z = 0): None places it by IK over the clip (candidates behind the wrist trajectory, on
    # the side of the head, `base_distance` m from its centre), else this (x, y) and yaw
    base_xy: tuple[float, float] | None = None
    base_yaw: float | None = None
    base_distance: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65)
    base_lateral: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)
    rest_q: tuple[float, ...] = (0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785)   # IK posture bias (Panda's ready pose)
    # PD (Genesis' Franka examples) and the Panda's torque limits
    kp: tuple[float, ...] = (4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0)
    kv: tuple[float, ...] = (450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0)
    effort: tuple[float, ...] = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    # action: residual on the joint reference (the clip's palm poses by IK) at t+1 (rad)
    action_scale: tuple[float, ...] = (0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1)
    # reward terms (rl_recon_env.py: arm_*)
    w_posture: float = 0.3                   # exp(-|q - q_ref| / sigma_posture): the IK posture (elbow)
    sigma_posture: float = 0.3
    w_limit: float = 0.5                     # joints within limit_margin of their limits
    limit_margin: float = 0.15
    w_velocity: float = 0.05                 # (dq / velocity limit)^2
    w_torque: float = 0.05                   # (torque / effort)^2
    w_table: float = 1.0                     # arm links below table_clearance (m) above the table
    table_clearance: float = 0.08            # link frames sit on the joint axes, ~6 cm inside the links' surface
    # DexTaG's AllDOFPowerPenalty: + w_power exp(-k_power sum |torque dq|) over the arm and finger joints. DexTaG's
    # k = 10 /W would read 0 here: the Panda tracking the reference spends ~50 W (median on pick_up_hammer/095941,
    # 300-550 W in its fast swing), the fingers ~0.2 W; w_power ~10% of the positive terms, as DexTaG's 50 / ~485
    w_power: float = 0.4
    k_power: float = 0.01                    # 1/W


class TactileArgs(BaseModel):
    """The simulated glove: the reference's taxels (Benchmark-Genesis' glove cells carried onto the Wuji links)."""
    model_config = frozen_config()
    mapping: Literal["layout", "anatomical", "mano"] = "mano"   # informational (the taxels are the reference's)
    assets_dir: str = "assets/tactile"
    probe_radius: float = 0.004              # contact if the taxel is closer than this to the surface
    kn: float = 250.0                        # simulated pressure (glove obs units) per metre of penetration:
                                             # a taxel 4 mm into the surface reads 1, like ~30 glove units
    soft_tau: float = 0.0015                 # soft contact temperature (Benchmark-Genesis)
    glove_scale: float = 30.0                # cleaned glove units per observation unit


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
    arm: ArmArgs = ArmArgs()                   # enabled: each hand on a Franka Panda (--env.arm.enabled=True)
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
    # the object's target: "dense", the reference's trajectory on every frame (held: the reference palm times one
    # offset); "keyframes", only its measured poses (obj_kf / obj_pos_measured / obj_rot_measured, prepare_reference.py
    # or scripts/rl/add_object_keyframes.py), within kf_window frames of a keyframe, margins x (1 + kf_window_margin x
    # the frames to it); the observation then gives the next keyframe's pose and the time to it, and the object
    # terminates an episode only there
    obj_reward: Literal["dense", "keyframes"] = "dense"
    kf_window: int = 3
    kf_window_margin: float = 0.5
    kf_rot_weight: float = 1.0                 # keyframes: the full rotation's hinge (measured there: 2-11 deg median
    kf_rot_margin: float = 0.35                # noise on the resting objects, 17 deg p90), off in "dense"
    kf_mask_axis: bool = True                  # keyframes: the SAM3 mask's axis still scores every frame it is valid
    # episodes. Training starts: "first" where evaluation starts (frame 0, or the first frame where the reference
    # hands clear the resting objects), to the clip end; "random": anywhere (the options below)
    train_start: Literal["first", "random"] = "first"
    start_frame_prob: float = 0.5              # train_start=random: random start where the objects rest (else frame 0)
    rest_starts: Literal["before_first_grasp", "away_from_holds"] = "before_first_grasp"   # rl_recon: the first
    rest_margin_s: float = 0.33                # ... at least this far from a grasp (rl_recon: 10 frames)
    rsi_grasp_prob: float = 0.5                # train_start=random: start at a grasp frame, the object in the hand
    rsi_min_contact_frac: float = 0.5          # ... where the glove fires >= this x the hand's peak
    rsi_min_clearance: float = 0.003           # ... and the held object clears the table (m)
    start_clearance: float = -1.0              # starts need the reference taxels this far from resting objects (m);
                                               # < 0: the reference's (this pipeline's: 5 mm; Benchmark-Genesis': off)
    episode_window_s: float = 0.0              # train_start=random: episodes last this long; 0: to the clip end
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


ENV_ARGS_CLASSES = {"RlReconEnv": RlReconEnvArgs}


def load_env_args(path) -> BaseModel:
    """A saved env configuration (configs/env_args.json), parsed with its environment's class."""
    import json
    from pathlib import Path

    text = Path(path).read_text()
    return ENV_ARGS_CLASSES[json.loads(text)["env_name"]].model_validate_json(text)
