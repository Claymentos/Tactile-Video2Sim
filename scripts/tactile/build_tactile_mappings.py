#!/usr/bin/env python3
"""Place the glove cells on both Wuji hands: layout, anatomical (tv2s_env/tactile/mapping.py)
and mano (glove -> MANO -> Wuji, tv2s_env/tactile/mano_mapping.py; also writes the MANO anchors).

HumanTouch writes assets/tactile/{layout,anatomical,mano}_{left,right}.npz (TaxelSet),
mano_anchors_{left,right}.npz, human_proportions.json and a check figure per side (the flat Wuji hand,
palmar view, taxels coloured by glove region). EgoTouch (--dataset egotouch, its layout
assets/tactile/egotouch_glove.json from extract_egotouch.py) only the dataset-agnostic one:
mano_egotouch_{left,right}.npz, mano_anchors_egotouch_{left,right}.npz, mappings_egotouch_{side}.png
(the env's --env.tactile.mapping=mano_egotouch).

    python scripts/tactile/build_tactile_mappings.py --data data/humantouch --dataset-root ~/HumanTouch/X001
    python scripts/tactile/build_tactile_mappings.py --dataset egotouch
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.glove import load_glove_layout  # noqa: E402
from tv2s_data.humantouch import SIDES, dataset_paths  # noqa: E402
from tv2s_env.robots.wuji_model import WujiHand  # noqa: E402
from tv2s_env.tactile.mano_mapping import humantouch_anchors, mano_to_wuji  # noqa: E402
from tv2s_env.tactile.mapping import (  # noqa: E402
    WujiSurface,
    build_anatomical_taxels,
    build_layout_taxels,
    human_hand_proportions,
)


def plot(hand, layout, taxel_sets, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surf = WujiSurface(hand)
    x, z = surf.radial, surf.normal
    y = np.cross(z, x)
    proj = lambda P: (P - surf.O) @ np.stack([x, y]).T
    T0 = surf.T0
    verts = np.concatenate([hand.collision_mesh(l).vertices @ T0[l][:3, :3].T + T0[l][:3, 3]
                            for l in hand.movable_links() if hand.collision_mesh(l) is not None])
    regions = sorted(set(layout.region))
    fig, axes = plt.subplots(1, len(taxel_sets), figsize=(6 * len(taxel_sets), 7))
    for ax, ts in zip(np.atleast_1d(axes), taxel_sets):
        world = np.array([T0[l][:3, :3] @ p + T0[l][:3, 3] for l, p in zip(ts.link, ts.pos)])
        V, W = proj(verts), proj(world)
        ax.scatter(V[::3, 0], V[::3, 1], s=1, color=(0.8, 0.8, 0.8))
        for i, r in enumerate(regions):
            m = layout.region[ts.channel] == r
            ax.scatter(W[m, 0], W[m, 1], s=6, color=plt.cm.tab10(i), label=r)
        ax.set_aspect("equal")
        ax.set_title(f"{ts.method}: {len(ts.channel)} taxels, {len(np.unique(ts.channel))}/{ts.n_channels} cells")
        ax.legend(fontsize=7)
    fig.suptitle(f"Wuji {hand.side} hand, palmar view")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("humantouch", "egotouch"), default="humantouch")
    ap.add_argument("--data", default=None, help="prepared HumanTouch folder (retarget/ gives the human proportions)")
    ap.add_argument("--dataset-root", default=None, help="HumanTouch task folder (its glove mapping)")
    ap.add_argument("--out", default=str(REPO / "assets" / "tactile"))
    ap.add_argument("--finger-tip-row", type=int, default=22)
    ap.add_argument("--points-per-cell", type=int, default=2, help="layout mapping: points per cell side")
    ap.add_argument("--angle-span-deg", type=float, default=140.0, help="arc of a finger patch around the phalanx")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.dataset == "egotouch":
        from tv2s_data.egotouch import LAYOUT_JSON, write_layout

        layout_json = LAYOUT_JSON if LAYOUT_JSON.exists() else write_layout()
        for side in SIDES:
            layout = load_glove_layout(layout_json, side)
            anchors = humantouch_anchors(layout, angle_span=np.radians(args.angle_span_deg))
            anchors.save(str(out / f"mano_anchors_egotouch_{side}.npz"))
            ts = mano_to_wuji(WujiHand(side), anchors)
            ts.method = "mano_egotouch"
            ts.save(str(out / f"mano_egotouch_{side}.npz"))
            print(f"mano_egotouch {side:5s}: {len(ts.channel)} taxels on {len(set(ts.link))} links, "
                  f"{len(np.unique(ts.channel))}/{ts.n_channels} cells covered")
            plot(WujiHand(side), layout, [ts], out / f"mappings_egotouch_{side}.png")
        print("->", out)
        return
    if not (args.data and args.dataset_root):
        raise SystemExit("--data and --dataset-root are needed for HumanTouch")
    mapping_json = dataset_paths(Path(args.dataset_root).expanduser())["tactile_mapping"]

    proportions = {}
    for side in SIDES:
        kps = [np.load(p)[f"{side}_keypoints"] for p in sorted((Path(args.data) / "retarget").glob("*.npz"))]
        if not kps:
            raise SystemExit("no retargeted episodes: run scripts/data/retarget_humantouch.py first")
        human = human_hand_proportions(np.concatenate(kps))
        proportions[side] = {"phalanges": {k: v.tolist() for k, v in human["phalanges"].items()},
                             "palm": {k: v.tolist() for k, v in human["palm"].items()}}
        layout = load_glove_layout(mapping_json, side, finger_tip_row=args.finger_tip_row)
        hand = WujiHand(side)
        span = np.radians(args.angle_span_deg)
        anchors = humantouch_anchors(layout, angle_span=span)          # glove -> MANO, then MANO -> Wuji
        anchors.save(str(out / f"mano_anchors_{side}.npz"))
        sets = [build_layout_taxels(hand, layout, points_per_cell=args.points_per_cell, angle_span=span),
                build_anatomical_taxels(hand, layout, human, angle_span=span),
                mano_to_wuji(hand, anchors)]
        for ts in sets:
            ts.save(str(out / f"{ts.method}_{side}.npz"))
            print(f"{ts.method:10s} {side:5s}: {len(ts.channel)} taxels on {len(set(ts.link))} links, "
                  f"{len(np.unique(ts.channel))}/{ts.n_channels} cells covered")
        plot(hand, layout, sets, out / f"mappings_{side}.png")
    with open(out / "human_proportions.json", "w") as f:
        json.dump(proportions, f, indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
