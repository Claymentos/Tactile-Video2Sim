from tv2s_agent.runners.config.schema import RunnerArgs

RunnerArgsRegistry = {
    # rl_recon: 1500 iterations x 32 steps x 2048 envs; checkpoints kept often, the eval script picks the best
    "recon": RunnerArgs(total_iterations=1500, log_interval=10, save_interval=25),
}
