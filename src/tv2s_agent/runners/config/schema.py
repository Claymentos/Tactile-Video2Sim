from pathlib import Path

from pydantic import BaseModel, ConfigDict


class RunnerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    total_iterations: int = 1500
    log_interval: int = 10
    save_interval: int = 250
    video_interval: int = 500          # a video of the current policy logged every this many iterations (0: off)
    save_path: Path = Path("./logs")
