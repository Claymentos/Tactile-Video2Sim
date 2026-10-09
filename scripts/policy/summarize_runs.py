#!/usr/bin/env python3
"""One table over several runs: the best evaluated checkpoint of each (by contact F1, then completion).

    python scripts/policy/summarize_runs.py EXP_NAME [EXP_NAME ...]
"""

import json
import sys

from common import REPO, latest_run_dir

COLS = ["success", "completion", "tac_f1", "tac_precision", "tac_recall", "tac_map_similarity", "e_wrist_cm",
        "e_tip_mm", "e_obj_cm"]


def main() -> None:
    print(f"{'run':<34} {'ckpt':<22} " + " ".join(f"{c[:10]:>10}" for c in COLS))
    for exp in sys.argv[1:]:
        evals = sorted((latest_run_dir(REPO / "logs" / exp) / "eval").glob("*.json"))
        if not evals:
            print(f"{exp:<34} (not evaluated)")
            continue
        rows = [(p.stem, json.loads(p.read_text())["summary"]) for p in evals]
        name, s = max(rows, key=lambda r: (round(r[1]["tac_f1"], 3), r[1]["completion"]))
        print(f"{exp:<34} {name:<22} " + " ".join(f"{s[c]:>10.3f}" for c in COLS))


if __name__ == "__main__":
    main()
