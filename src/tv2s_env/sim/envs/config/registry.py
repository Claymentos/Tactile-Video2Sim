"""Named environment configurations.

  rl_recon   Benchmark-Genesis' rl_recon (its env.py) on a per-episode reference.npz (scripts/rl/prepare_reference.py),
             the hands the demonstration uses (rl_recon_env.py)
"""

from tv2s_env.sim.envs.config.schema import RlReconEnvArgs

EnvArgsRegistry = {"rl_recon": RlReconEnvArgs()}
