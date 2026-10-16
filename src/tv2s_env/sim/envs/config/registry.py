"""Named environment configurations (DexTaG's EnvArgsRegistry).

  bimanual_dextag   DexTaG's retargeter: DexTaG's rewards and observations, `layout` glove mapping
  bimanual_recon    Benchmark-Genesis' rl_recon: residual policy, contact-F1 reward, `anatomical` mapping
  rl_recon          Benchmark-Genesis' rl_recon as its env.py, on a per-episode reference.npz
                    (scripts/rl/prepare_reference.py), the hands the demonstration uses (rl_recon_env.py)
The `*_<mapping>` variants swap the glove mapping only, to compare the mappings under one method
(`mano`: glove -> MANO -> Wuji, tactile/mano_mapping.py).
"""

from tv2s_env.sim.envs.config.schema import (
    DexTaGEnvArgs,
    ObjectArgs,
    ReconEnvArgs,
    RlReconEnvArgs,
    TactileArgs,
)

# === DexTaG observation terms (per hand, both hands concatenated) ===
PROPRIO_TERMS = [
    "hand_dof_pos",                 # 20
    "hand_dof_vel",                 # 20
    "finger_link_pos",              # 15
    "finger_link_quat",             # 20
    "base_pos",                     # 3
    "base_quat",                    # 4
    "base_lin_vel",                 # 3
    "base_ang_vel",                 # 3
    "object_pos_rel",               # 3
    "fingertip_to_surface_dist",    # 5
    "object_quat",                  # 4
    "object_lin_vel",               # 3
    "object_ang_vel",               # 3
    "base_dof_pos",                 # 6  (DexTaG: arm_dof_pos, 7)
    "base_dof_vel",                 # 6
    "support_contact_force",        # 1  (DexTaG: table_contact_force_magnitude)
]  # 119 per hand

TARGET_MOTION_TERMS = [
    "target_hand_dof_pos",          # 20
    "target_hand_dof_vel",          # 20
    "delta_fingertip_pos",          # 15
    "target_fingertip_quat",        # 20
    "delta_wrist_pos",              # 3
    "target_wrist_quat",            # 4
    "target_wrist_vel",             # 3
    "target_wrist_ang_vel",         # 3
    "delta_object_pos",             # 3
    "target_object_quat",           # 4
    "target_object_vel",            # 3
    "target_object_ang_vel",        # 3
    "object_valid",                 # 1  (a hand may have no object to track)
]  # 102 per hand

OBS_SCALES = {
    "object_pos_rel": 5.0, "fingertip_to_surface_dist": 10.0, "object_ang_vel": 0.1,
    "hand_dof_vel": 0.1, "base_dof_vel": 0.1, "target_hand_dof_vel": 0.1,
    "delta_fingertip_pos": 10.0, "delta_wrist_pos": 10.0, "target_wrist_vel": 1.0,
    "delta_object_pos": 10.0, "target_object_vel": 1.0, "support_contact_force": 0.05,
    "tactile_map": 1.0,
}

FINGER_WEIGHTS = dict(thumb_weight=0.8, index_weight=0.8, middle_weight=0.8, ring_weight=0.6, pinky_weight=0.6)

EnvArgsRegistry: dict = {}

EnvArgsRegistry["bimanual_dextag"] = DexTaGEnvArgs(
    env_name="DexTaGBimanualEnv",
    tactile=TactileArgs(mapping="layout"),
    objects=ObjectArgs(
        mass_randomization_range=0.3, friction_range=(0.7, 1.5),
        init_pos_noise_xy=0.02, init_rot_noise_deg=10.0,
        perturbation_force_scale=20.0, perturbation_torque_scale=0.2,
    ),
    proprio_terms=PROPRIO_TERMS,
    target_motion_terms=TARGET_MOTION_TERMS,
    obs_scales=OBS_SCALES,
    # proprio_terms are always observed (the encoder's proprio slot)
    actor_obs_terms=["target_motion", "tactile_map", "target_action_flat"],
    critic_obs_terms=["target_motion", "last_reward", "tactile_map", "target_tactile_map",
                      "action_proprio_history_flat", "target_action_flat"],
    reward_args={
        # trajectory tracking
        "WristPositionTrackingReward": {"scale": 30.0, "k": 40.0},
        "WristRotationTrackingReward": {"scale": 45.0, "k": 2.0},
        "FingertipPositionTrackingReward": {"scale": 50.0, **FINGER_WEIGHTS, "thumb_k": 100.0, "index_k": 100.0,
                                            "middle_k": 100.0, "ring_k": 60.0, "pinky_k": 60.0},
        "FingertipRotationTrackingReward": {"scale": 35.0, **FINGER_WEIGHTS},
        "HandDofPositionTrackingReward": {"scale": 25.0, "k": 1.0},
        "ObjectPositionTrackingReward": {"scale": 100.0, "k": 40.0},
        "ObjectRotationTrackingReward": {"scale": 125.0, "k": 4.0},
        # tactile as guidance
        "WeightedGaussianBlurredTactileMapSimilarityReward": {"scale": 50.0, "k": 4.0},
        "TactileGatedFingertipSurfaceProximityReward": {"scale": 25.0, "k": 50.0},
        # contact and smoothness
        "SupportContactPenalty": {"scale": 50.0, "k": 0.005, "threshold": 0.0},
        "ObjectVelocityChangePenalty": {"scale": 50.0, "k": 2.0, "threshold": 0.1, "height_threshold": 0.05},
        "ObjectAngularVelocityChangePenalty": {"scale": 50.0, "k": 0.5, "threshold": 3.0, "height_threshold": 0.05},
        "AllDOFPowerPenalty": {"scale": 50.0, "k": 0.05},
    },
)
EnvArgsRegistry["bimanual_recon"] = ReconEnvArgs(
    env_name="ReconBimanualEnv",
    tactile=TactileArgs(mapping="anatomical"),
    reward_args={
        "TactileContactF1Reward": {"scale": 1.5, "probe_radius": 0.004, "soft_tau": 0.0015, "reach_weight": 0.5},
        "HandTrackingKernelReward": {"scale": 1.0, "sigma_pos": 0.02, "sigma_rot": 0.25, "sigma_kp": 0.03},
        "ObjectPositionHingeReward": {"scale": 0.5},
        "ObjectAxisHingeReward": {"scale": 0.75},
        "ActionRatePenalty": {"scale": 0.02},
    },
)

# mapping swaps, for the method x mapping comparison
for _name, _mapping in [("bimanual_dextag_anatomical", "anatomical"), ("bimanual_recon_layout", "layout"),
                         ("bimanual_dextag_mano", "mano"), ("bimanual_recon_mano", "mano")]:
    _base = EnvArgsRegistry[_name.rsplit("_", 1)[0]]
    EnvArgsRegistry[_name] = _base.model_copy(update={"tactile": _base.tactile.model_copy(update={"mapping": _mapping})})

EnvArgsRegistry["rl_recon"] = RlReconEnvArgs()
