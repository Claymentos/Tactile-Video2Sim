#!/usr/bin/env python3
"""Download EgoTouch episodes (huggingface: zhouzhoujy/EgoTouch) into --root, skipping files already there.

An episode is ~2 MB (320x240 videos); the dataset ~2 500 episodes over 213 tasks. Selection: --tasks (task
names, or scene/task), at most --per-task episodes of each, or --episodes (recording times or their prefixes);
--list prints the tasks and their episode counts. Only the files the pipeline reads are fetched
(tv2s_data.egotouch.FILES; --files to change). Plain HTTPS, no token needed (the dataset is public).

    python scripts/data/download_egotouch.py --root ~/EgoTouch --tasks pick_up_bottle,twist_bottle_cap --per-task 5
    python scripts/data/download_egotouch.py --list
"""

import argparse
import collections
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.egotouch import FILES, REPO_ID  # noqa: E402

API = f"https://huggingface.co/api/datasets/{REPO_ID}"
RESOLVE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"


def remote_episodes() -> dict[str, list[str]]:
    """scene/task/stamp -> its files, from the dataset's file list."""
    with urllib.request.urlopen(API, timeout=60) as r:
        names = [s["rfilename"] for s in json.load(r)["siblings"]]
    eps = collections.defaultdict(list)
    for n in names:
        parts = n.split("/")
        if len(parts) == 4:
            eps["/".join(parts[:3])].append(parts[3])
    return dict(sorted(eps.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path.home() / "EgoTouch"))
    ap.add_argument("--tasks", default=None, help="comma list of task names (or scene/task)")
    ap.add_argument("--episodes", default=None, help="comma list of recording times (or prefixes)")
    ap.add_argument("--per-task", type=int, default=5, help="episodes per task (0: all)")
    ap.add_argument("--files", default=",".join(FILES))
    ap.add_argument("--list", action="store_true", help="print the tasks and their episode counts")
    args = ap.parse_args()
    eps = remote_episodes()
    if args.list:
        for task, n in sorted(collections.Counter(e.rsplit("/", 1)[0] for e in eps).items()):
            print(f"{n:4d}  {task}")
        print(f"{len(eps)} episodes")
        return
    if not (args.tasks or args.episodes):
        raise SystemExit("give --tasks or --episodes (--list shows the tasks)")
    tasks = [t for t in (args.tasks or "").split(",") if t]
    stamps = [s for s in (args.episodes or "").split(",") if s]
    chosen, per = [], collections.Counter()
    for e in eps:
        scene, task, stamp = e.split("/")
        if tasks and not any(t in (task, f"{scene}/{task}") for t in tasks):
            continue
        if stamps and not any(stamp.startswith(s) for s in stamps):
            continue
        if args.per_task and per[task] >= args.per_task:
            continue
        per[task] += 1
        chosen.append(e)
    wanted = [f for f in args.files.split(",") if f]
    root = Path(args.root).expanduser()
    for i, e in enumerate(chosen):
        d = root / e
        d.mkdir(parents=True, exist_ok=True)
        got = 0
        for f in wanted:
            if f not in eps[e] or (d / f).exists():
                continue
            tmp = d / f"{f}.part"
            urllib.request.urlretrieve(f"{RESOLVE}/{e}/{f}", tmp)
            tmp.rename(d / f)
            got += 1
        missing = [f for f in wanted if f not in eps[e]]
        print(f"[{i + 1}/{len(chosen)}] {e}: {got} files" + (f" (not in the dataset: {', '.join(missing)})" if missing else ""),
              flush=True)
    print(f"{len(chosen)} episodes -> {root}")


if __name__ == "__main__":
    main()
