from tv2s_agent.runners.config.schema import RunnerArgs

RunnerArgsRegistry = {
    # DexTaG trains its retargeter for 15k iterations of 32 steps x 3072 envs
    "dextag": RunnerArgs(total_iterations=15000, log_interval=10, save_interval=500),
    # rl_recon: 1500 iterations x 32 steps x 2048 envs; checkpoints kept often, the eval script picks the best
    "recon": RunnerArgs(total_iterations=1500, log_interval=10, save_interval=25),
}
