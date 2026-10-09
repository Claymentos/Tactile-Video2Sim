#!/usr/bin/env python3
"""Compare the two ways of mapping the HumanTouch glove onto the Wuji Hand (DexTaG vs Benchmark-Genesis).

Each repo has (a) a placement of the glove cells on the robot and (b) a tactile signal + reward:
  DexTaG              layout placement (dense points binned by a 2D layout)  +  pressure image, compared with
                      the normalised / blurred / contact-masked image similarity
  Benchmark-Genesis   anatomical placement (one taxel per cell, through the human hand's proportions)  +
                      binary contact per channel (distance < probe radius), compared with soft contact F1

The comparison crosses the two placements with the two signal models, without any policy, on
"kinematic replay": the Wuji hands are put on the retargeted reference (forward kinematics), the proxy
object on its reference pose, and the simulated glove is read and compared with the real one.

  1. geometry       where each glove row / region lands on the Wuji, and how far apart the two placements put
                    the same cell
  2. agreement      contact precision / recall / F1 (several probe radii, since the replayed hand is only
                    approximately on the proxy's surface) and image similarity, per region too
  3. sensitivity    both rewards under a displaced object (0-8 cm): a useful reward peaks at the reference

Writes <out>/comparison.json and <out>/comparison.md.

    python scripts/tactile/compare_tactile_mappings.py --clips data/humantouch/clips --out outputs/tactile_comparison
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.glove import REGIONS, load_glove_layout  # noqa: E402
from tv2s_data.humantouch import SIDES  # noqa: E402
from tv2s_env.common.rewards.dextag_terms import WeightedGaussianBlurredTactileMapSimilarityReward  # noqa: E402
from tv2s_env.common.rewards.recon_terms import TactileContactF1Reward  # noqa: E402
from tv2s_env.robots.wuji_model import WujiHand  # noqa: E402
from tv2s_env.tactile.mapping import GLOVE_TO_WUJI_FINGER, TaxelSet, WujiSurface  # noqa: E402
from tv2s_env.tactile.sensor import GloveSensor  # noqa: E402

METHODS = ("layout", "anatomical")
PROBE_RADII = (0.004, 0.008, 0.015)


# ---------------------------------------------------------------- 1. geometry
def geometry_report(hand: WujiHand, layout, sets: dict[str, TaxelSet]) -> dict:
    surf = WujiSurface(hand)
    world = {m: np.array([surf.T0[l][:3, :3] @ p + surf.T0[l][:3, 3] for l, p in zip(ts.link, ts.pos)])
             for m, ts in sets.items()}
    out = {"taxels": {m: int(len(ts.channel)) for m, ts in sets.items()}, "rows_to_links": {}, "displacement_mm": {}}
    # finger rows -> Wuji phalanx (majority link of the row's taxels), per mapping
    for region in GLOVE_TO_WUJI_FINGER:
        m_reg = layout.region == region
        rows = sorted(set(layout.rows[m_reg]))
        out["rows_to_links"][region] = {}
        for m, ts in sets.items():
            seq = []
            for r in rows:
                chans = np.flatnonzero(m_reg & (layout.rows == r))
                links = ts.link[np.isin(ts.channel, chans)]
                seq.append(max(set(links), key=list(links).count).split("_")[-1])
            out["rows_to_links"][region][m] = dict(zip([int(r) for r in rows], seq))
    # same cell, two placements: centroid of the layout points vs the anatomical taxel
    cent = {m: np.stack([world[m][ts.channel == c].mean(0) for c in range(ts.n_channels)]) for m, ts in sets.items()}
    d = np.linalg.norm(cent["layout"] - cent["anatomical"], axis=1) * 1000
    for region in REGIONS:
        m = layout.region == region
        out["displacement_mm"][region] = dict(median=float(np.median(d[m])), p90=float(np.percentile(d[m], 90)))
    return out


# ---------------------------------------------------------------- 2./3. kinematic replay
class ReplayGlove:
    """The simulated glove of one hand, read on the reference hand (forward kinematics, no physics)."""

    def __init__(self, hand: WujiHand, layout, taxels: TaxelSet, device: torch.device):
        self.hand = hand
        self.links = sorted(set(taxels.link))
        self.sensor = GloveSensor(taxels, layout, {l: i for i, l in enumerate(self.links)}, probe_radius=0.004,
                                  kn=250.0, device=device)
        self.device = device

    def link_poses(self, qpos: np.ndarray, palm_pos: np.ndarray, palm_quat: np.ndarray):
        base = np.tile(np.eye(4), (len(qpos), 1, 1))
        base[:, :3, :3] = Rotation.from_quat(palm_quat[:, [1, 2, 3, 0]]).as_matrix()
        base[:, :3, 3] = palm_pos
        fk = self.hand.fk(qpos, base)
        T = np.stack([fk[l] for l in self.links], 1)                                  # (N, L, 4, 4)
        f = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return f(T[..., :3, 3]), f(T[..., :3, :3])

    def read(self, link_pos, link_rot, obj_pos, obj_rot, radius, half_length):
        return self.sensor.read(link_pos, link_rot, obj_pos, obj_rot, radius, half_length)


def replay_frames(clips: list[dict], side: str, stride: int) -> dict[str, np.ndarray]:
    """Frames where the hand holds its object and the glove reads contact."""
    rows = {k: [] for k in ["qpos", "palm_pos", "palm_quat", "obj", "radius", "half", "pressure", "contact"]}
    for c in clips:
        h = c["hands"][side]
        o = h["object"]
        if o is None:
            continue
        t0, t1 = o["grasp_interval"]
        for t in range(t0, t1, stride):
            if h["tactile_contact"][t].sum() == 0:
                continue
            rows["qpos"].append(h["dof_positions"][t]), rows["palm_pos"].append(h["wrist_positions"][t])
            rows["palm_quat"].append(h["wrist_quat"][t]), rows["obj"].append(o["pose_matrices"][t])
            rows["radius"].append(o["radius"]), rows["half"].append(o["length"] / 2)
            rows["pressure"].append(h["tactile_pressure"][t]), rows["contact"].append(h["tactile_contact"][t])
    return {k: np.asarray(v) for k, v in rows.items()}


def prf(sim: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    tp = (sim & gt).sum().item()
    p, r = tp / max(sim.sum().item(), 1), tp / max(gt.sum().item(), 1)
    return dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9))


def agreement(glove: ReplayGlove, fr: dict, layout, glove_scale: float, device) -> dict:
    f = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    lp, lr = glove.link_poses(fr["qpos"], fr["palm_pos"], fr["palm_quat"])
    obj = f(fr["obj"])
    dist, pressure = glove.read(lp, lr, obj[:, :3, 3], obj[:, :3, :3], f(fr["radius"]), f(fr["half"]))
    gt = torch.as_tensor(fr["contact"], device=device)
    out = {"frames": int(len(gt)), "contact": {}, "regions": {}}
    for r in PROBE_RADII:
        out["contact"][f"{r * 1000:.0f}mm"] = prf(dist < r, gt)
    # region level: is the region in contact at all (palm, each finger)
    for name in REGIONS:
        m = torch.as_tensor(layout.region == name, device=device)
        out["regions"][name] = prf((dist[:, m] < PROBE_RADII[0]).any(-1), gt[:, m].any(-1))
    # DexTaG's image similarity (reference pressure on contact cells, simulated pressure)
    sim_term = WeightedGaussianBlurredTactileMapSimilarityReward()
    tgt = glove.sensor.to_image(f(fr["pressure"]) / glove_scale * gt)
    cur = glove.sensor.to_image(pressure)
    w = glove.sensor.to_image(glove.sensor.channel_weights[None])[0][None]
    out["map_similarity"] = float(sim_term.similarity(cur[:, None], tgt[:, None], w).mean())
    out["soft_f1_reward"] = float(TactileContactF1Reward()._compute(dist[:, None], gt[:, None]).mean())
    return out


def sensitivity(glove: ReplayGlove, fr: dict, glove_scale: float, device, offsets=(0.0, 0.01, 0.02, 0.04, 0.08),
                n_dirs: int = 6, seed: int = 0) -> dict:
    """Both tactile rewards with the object displaced by `offsets` metres in random directions."""
    rng = np.random.default_rng(seed)
    f = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    lp, lr = glove.link_poses(fr["qpos"], fr["palm_pos"], fr["palm_quat"])
    obj = f(fr["obj"])
    gt = torch.as_tensor(fr["contact"], device=device)
    tgt = glove.sensor.to_image(f(fr["pressure"]) / glove_scale * gt)
    w = glove.sensor.to_image(glove.sensor.channel_weights[None])[0][None]
    sim_term, f1_term = WeightedGaussianBlurredTactileMapSimilarityReward(), TactileContactF1Reward()
    out = {"offset_cm": [100 * o for o in offsets], "map_similarity": [], "soft_f1_reward": []}
    dirs = rng.normal(size=(n_dirs, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for o in offsets:
        ms, fs = [], []
        for d in dirs if o > 0 else dirs[:1]:
            dist, pressure = glove.read(lp, lr, obj[:, :3, 3] + f(o * d), obj[:, :3, :3], f(fr["radius"]), f(fr["half"]))
            ms.append(float(sim_term.similarity(glove.sensor.to_image(pressure)[:, None], tgt[:, None], w).mean()))
            fs.append(float(f1_term._compute(dist[:, None], gt[:, None]).mean()))
        out["map_similarity"].append(float(np.mean(ms)))
        out["soft_f1_reward"].append(float(np.mean(fs)))
    return out


# ---------------------------------------------------------------- report
def markdown(res: dict) -> str:
    L = ["# Glove -> Wuji tactile mapping: DexTaG (layout) vs Benchmark-Genesis (anatomical)", "",
         "Generated by `scripts/tactile/compare_tactile_mappings.py` on kinematic replays of the reference "
         "(no policy). See docs/METHODS.md for the discussion.", ""]
    for side in SIDES:
        r = res[side]
        g = r["geometry"]
        L += [f"## {side} hand", "", f"Taxels: layout {g['taxels']['layout']}, anatomical {g['taxels']['anatomical']} "
              f"(360 glove cells).", "", "Finger patch rows (proximal -> tip) -> Wuji phalanx:", "",
              "| finger | layout | anatomical |", "|---|---|---|"]
        for region, d in g["rows_to_links"].items():
            L.append(f"| {region} | {' '.join(d['layout'].values())} | {' '.join(d['anatomical'].values())} |")
        L += ["", "Distance between the two placements of the same cell (mm):", "", "| region | median | p90 |", "|---|---|---|"]
        for region, d in g["displacement_mm"].items():
            L.append(f"| {region} | {d['median']:.1f} | {d['p90']:.1f} |")
        L += ["", f"Kinematic replay on {r['agreement']['layout']['frames']} grasp frames:", "",
              "| mapping | " + " | ".join(f"F1 @{k}" for k in r["agreement"]["layout"]["contact"]) +
              " | P @4mm | R @4mm | image similarity (DexTaG) | soft F1 + reach (rl_recon) |",
              "|---|" + "---|" * (len(PROBE_RADII) + 4)]
        for m in METHODS:
            a = r["agreement"][m]
            c = a["contact"]
            L.append(f"| {m} | " + " | ".join(f"{v['f1']:.3f}" for v in c.values()) +
                     f" | {c['4mm']['precision']:.3f} | {c['4mm']['recall']:.3f} | {a['map_similarity']:.3f} | "
                     f"{a['soft_f1_reward']:.3f} |")
        L += ["", "Region-level contact F1 (is the region touching at all):", "",
              "| mapping | " + " | ".join(REGIONS) + " |", "|---|" + "---|" * len(REGIONS)]
        for m in METHODS:
            L.append(f"| {m} | " + " | ".join(f"{r['agreement'][m]['regions'][x]['f1']:.2f}" for x in REGIONS) + " |")
        L += ["", "Reward under a displaced object (mean over frames and 6 directions):", "",
              "| mapping | reward | " + " | ".join(f"{o:g} cm" for o in r["sensitivity"]["layout"]["offset_cm"]) + " |",
              "|---|---|" + "---|" * len(r["sensitivity"]["layout"]["offset_cm"])]
        for m in METHODS:
            s = r["sensitivity"][m]
            L.append(f"| {m} | image similarity | " + " | ".join(f"{v:.3f}" for v in s["map_similarity"]) + " |")
            L.append(f"| {m} | soft F1 + reach | " + " | ".join(f"{v:.3f}" for v in s["soft_f1_reward"]) + " |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default=str(REPO / "data" / "humantouch" / "clips"))
    ap.add_argument("--assets", default=str(REPO / "assets" / "tactile"))
    ap.add_argument("--out", default=str(REPO / "outputs" / "tactile_comparison"))
    ap.add_argument("--stride", type=int, default=2, help="use every n-th grasp frame")
    ap.add_argument("--glove-scale", type=float, default=30.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    clip_paths = sorted(Path(args.clips).glob("*.pkl"))
    clips = [pickle.load(open(p, "rb")) for p in clip_paths]
    meta = clips[0]["metadata"]
    res = {"clips": len(clips)}
    for side in SIDES:
        layout = load_glove_layout(meta["tactile_mapping"], side, finger_tip_row=meta["finger_tip_row"])
        hand = WujiHand(side)
        sets = {m: TaxelSet.load(str(Path(args.assets) / f"{m}_{side}.npz")) for m in METHODS}
        fr = replay_frames(clips, side, args.stride)
        r = {"geometry": geometry_report(hand, layout, sets), "agreement": {}, "sensitivity": {}}
        for m in METHODS:
            glove = ReplayGlove(hand, layout, sets[m], device)
            r["agreement"][m] = agreement(glove, fr, layout, args.glove_scale, device)
            r["sensitivity"][m] = sensitivity(glove, fr, args.glove_scale, device)
            a = r["agreement"][m]
            print(f"{side:5s} {m:10s} frames {a['frames']:5d} | F1 " +
                  " ".join(f"{k} {v['f1']:.3f}" for k, v in a["contact"].items()) +
                  f" | image sim {a['map_similarity']:.3f} | soft F1 {a['soft_f1_reward']:.3f}")
        res[side] = r
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(res, indent=1))
    (out / "comparison.md").write_text(markdown(res))
    print("->", out / "comparison.md")


if __name__ == "__main__":
    main()
