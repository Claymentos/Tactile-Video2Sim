#!/usr/bin/env python3
"""What one EgoTouch episode's glove says, on one sheet: <out>/<key>_<side>_glove.png, and a summary on stdout.

  layout     the 21x21 grid (tv2s_data.egotouch.glove_cells): taxels by region, bend strips, cells off the glove;
             circled, the taxels the simulated glove carries onto the Wuji (assets/tactile/wuji_taxels_<side>.npz)
  snapshots  the grid (raw units: pressure_grids.npz x tactile_max) and the chest video at the first frame, the
             grasp onset, mid-hold, the frame of most contact and the release
  time       per region the strongest taxel, per finger its bend sensor, the holds shaded (tv2s_data.reference.
             contact_holds on the taxels above --threshold), the pipeline's threshold (recon_reference.
             load_real_channels: 5 raw) and TouchAnything's (egotouch.CONTACT_THRESHOLD x tactile_max)

Summary: per region the cells, their span along the finger (u: 0 proximal, 1 tip), how many the Wuji carries, the share
of the holds they fire; the cells firing before the first hold (offsets or bending, not the object); the dataset's
manual_contact_annotation.json.

Run in the `genesis` env:
    python scripts/data/inspect_egotouch_glove.py --key pick_up_hammer__20260321_095958_211 [--side left]
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.egotouch import CONTACT_THRESHOLD, GRID, episode_from_key, glove_cells  # noqa: E402
from tv2s_data.reference import contact_holds  # noqa: E402

REGIONS = ("thumb", "index", "middle", "ring", "pinky", "palm")
COLORS = dict(thumb="tab:red", index="tab:orange", middle="tab:olive", ring="tab:green", pinky="tab:cyan", palm="tab:blue")


def chest_frames(video: Path, frames: list[int]) -> dict[int, np.ndarray]:
    import cv2

    cap, out = cv2.VideoCapture(str(video)), {}
    for t in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, img = cap.read()
        if ok:
            out[t] = img[..., ::-1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True, help="EgoTouch key, e.g. pick_up_hammer__20260321_095958_211")
    ap.add_argument("--root", default="~/EgoTouch")
    ap.add_argument("--side", choices=("left", "right"), default=None, help="default: the glove that fires most")
    ap.add_argument("--threshold", type=float, default=5.0, help="contact, raw units (the pipeline's)")
    ap.add_argument("--out", default=str(REPO / "outputs" / "glove"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = episode_from_key(args.root, args.key)
    z = np.load(ep.dir / "pressure_grids.npz")
    scale = float(z["tactile_max"])
    raw = {s: z[f"{s}_pressure_grid"].reshape(len(z[f"{s}_pressure_grid"]), -1) * scale for s in ("left", "right")}
    side = args.side or max(raw, key=lambda s: np.nansum(raw[s] > args.threshold))
    G, T = raw[side], len(raw[side])
    cells, bend = glove_cells(side)
    region = {c[0]: c[3] for c in cells}
    u = {c[0]: c[4] for c in cells}
    taxels = np.load(REPO / "assets" / "tactile" / f"wuji_taxels_{side}.npz")
    carried = {int(r * GRID[1] + c) for r, c in taxels["cell_rc"]}
    ids = np.array(sorted(region))
    contact = G[:, ids] > args.threshold
    holds = contact_holds(contact)
    held = np.zeros(T, bool)
    for a, b in holds:
        held[a:b] = True

    # ---------------------------------------------------------------- summary
    print(f"{ep.key}, {side} glove: {T} frames, tactile_max {scale:g}, holds {holds}")
    print(f"{'region':7s} {'cells':>5s} {'u (0 base, 1 tip)':>18s} {'on the Wuji':>11s} {'fire in holds':>13s} {'max':>5s}")
    for r in REGIONS:
        rc = [i for i in ids if region[i] == r]
        fire = (G[:, rc] > args.threshold).any(1)
        us = [u[i] for i in rc]
        print(f"{r:7s} {len(rc):5d} {min(us):8.2f} .. {max(us):.2f} {len(carried & set(rc)):11d} "
              f"{fire[held].mean() if held.any() else 0:13.0%} {np.nanmax(G[:, rc]):5.0f}")
    pre = slice(0, holds[0][0] if holds else T)
    early = np.flatnonzero(np.nanmax(G[pre], 0) > args.threshold)
    strips = set(np.flatnonzero(np.isfinite(G[0]))) - set(region)
    kinds = collections.Counter("bend strip" if i in strips else region.get(i, "off the glove") for i in early)
    print(f"cells above {args.threshold:g} before the first hold (frames {pre.start}-{pre.stop - 1}): {dict(kinds)}")
    print(f"bend sensors (raw): " + ", ".join(f"{f} {np.nanmin(G[:, i]):.0f}-{np.nanmax(G[:, i]):.0f}" for f, i in bend[::3]))
    ann = ep.dir / "manual_contact_annotation.json"
    if ann.exists():
        a = json.loads(ann.read_text())
        print("manual_contact_annotation.json:", {k: v for k, v in a.items() if k.endswith("contact")})

    # ---------------------------------------------------------------- sheet
    n_tot = contact.sum(1)
    snaps = {"first frame": 0}
    if holds:
        a, b = holds[0]
        snaps.update({"grasp onset": a, "mid-hold": (a + b) // 2, "most contact": int(np.argmax(n_tot)), "release": min(b, T - 1)})
    imgs = chest_frames(ep.dir / "chest.mp4", list(snaps.values()))
    fig = plt.figure(figsize=(3.2 * (len(snaps) + 1), 13))
    gs = fig.add_gridspec(3, len(snaps) + 1, height_ratios=[1, 1, 1.1])

    ax = fig.add_subplot(gs[0:2, 0])
    lay = np.full(GRID, np.nan)
    for i in ids:
        lay.flat[i] = REGIONS.index(region[i])
    from matplotlib.colors import ListedColormap
    ax.imshow(lay, cmap=ListedColormap([COLORS[r] for r in REGIONS]), vmin=-0.5, vmax=5.5)
    for i in strips:
        ax.add_patch(plt.Rectangle((i % GRID[1] - 0.5, i // GRID[1] - 0.5), 1, 1, color="0.7"))
    for i in carried:
        ax.plot(i % GRID[1], i // GRID[1], "o", mfc="none", mec="k", ms=6)
    ax.set_title(f"layout ({side})\ncolour: region, grey: bend strip\no: carried onto the Wuji", fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])

    for k, (name, t) in enumerate(snaps.items()):
        ax = fig.add_subplot(gs[0, k + 1])
        if t in imgs:
            ax.imshow(imgs[t])
        ax.set_title(f"{name}: frame {t}", fontsize=9)
        ax.axis("off")
        ax = fig.add_subplot(gs[1, k + 1])
        ax.imshow(G[t].reshape(GRID), cmap="magma", vmin=0, vmax=scale)
        ax.set_title(f"{int(n_tot[t])} taxels > {args.threshold:g}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])

    ax = fig.add_subplot(gs[2, :])
    for r in REGIONS:
        rc = [i for i in ids if region[i] == r]
        ax.plot(np.nanmax(G[:, rc], 1), color=COLORS[r], label=r)
    for f, i in bend[1::3]:
        ax.plot(G[:, i], color=COLORS[f], ls=":", lw=1)
    for a, b in holds:
        ax.axvspan(a, b, color="0.9", zorder=0)
    ax.axhline(args.threshold, color="k", lw=0.8, ls="--", label=f"pipeline threshold ({args.threshold:g})")
    ax.axhline(CONTACT_THRESHOLD * scale, color="k", lw=0.8, ls="-.", label=f"TouchAnything ({CONTACT_THRESHOLD * scale:.1f})")
    ax.set_xlabel("frame (30 Hz)"), ax.set_ylabel("raw units")
    ax.set_title("solid: strongest taxel per region, dotted: bend sensor per finger, grey: holds", fontsize=9)
    ax.legend(fontsize=8, ncol=4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{ep.key}_{side}_glove.png"
    fig.tight_layout()
    fig.savefig(png, dpi=80)
    print(f"-> {png}")


if __name__ == "__main__":
    main()
