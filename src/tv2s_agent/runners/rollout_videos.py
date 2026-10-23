"""Videos of the current policy during training: every `RunnerArgs.video_interval` iterations, the checkpoint just
saved is replayed on the whole demonstration by scripts/policy/visualize_episode.py in a subprocess (one env,
offscreen rendering), so the training does not wait for it; the finished mp4 is logged to wandb at the iteration of
its checkpoint (key rollout/video) and kept in <run>/videos/.
"""

import subprocess
import sys
from pathlib import Path


class RolloutVideos:
    def __init__(self, script: Path, exp_name: str, run_dir: Path, interval: int, start: str = "first") -> None:
        self.script, self.exp_name, self.run_dir = Path(script), exp_name, Path(run_dir)
        self.interval, self.start = interval, start
        self.out = self.run_dir / "videos"
        self._pending: list[tuple[int, Path, subprocess.Popen]] = []

    def due(self, it: int) -> bool:
        """After iteration `it` (0-based): a video of checkpoint it + 1."""
        return self.interval > 0 and (it + 1) % self.interval == 0

    def launch(self, it: int, checkpoint: Path) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        checkpoint = Path(checkpoint)
        # visualize_episode.py's tag: <exp>_<start>_<checkpoint stem>
        mp4 = self.out / f"{self.exp_name}_{self.start}_{checkpoint.stem}.mp4"
        log = open(self.out / f"{checkpoint.stem}.log", "w")
        proc = subprocess.Popen([sys.executable, str(self.script), "--exp", self.exp_name, "--run", str(self.run_dir),
                                 "--checkpoint", str(checkpoint), "--start", self.start, "--out", str(self.out)],
                                stdout=log, stderr=subprocess.STDOUT)
        log.close()
        self._pending.append((it, mp4, proc))
        print(f"Rollout video of {checkpoint.name}: rendering in the background (pid {proc.pid}) -> {mp4}")

    def finished(self, wait: bool = False) -> list[tuple[int, Path]]:
        """The videos rendered since the last call, as (iteration, mp4); `wait`: wait for the running ones."""
        done, still = [], []
        for it, mp4, proc in self._pending:
            if wait:
                proc.wait()
            if proc.poll() is None:
                still.append((it, mp4, proc))
            elif proc.returncode == 0 and mp4.exists():
                done.append((it, mp4))
            else:
                print(f"Rollout video {mp4.name} failed (exit {proc.returncode}): see {self.out}")
        self._pending = still
        return done
