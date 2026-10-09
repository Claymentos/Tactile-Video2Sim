"""Put the HumanTouch glove cells on the Wuji Hand: the two methods compared in this repo.

Both produce a `TaxelSet`: points on the Wuji collision hulls (link frame), each assigned to one glove
channel (an assigned cell of tv2s_data.glove.GloveLayout). Both use the glove layout's normalised cell
coordinates (u: proximal -> distal, v: ulnar -> radial) and the same palmar/radial directions; they
differ in how a cell is placed on the robot.

layout  (DexTaG, its Genesis fork's examples/tactile pipeline: a 2D layout per link, then pixel pairing)
    Points are sampled densely on the palmar side of every Wuji link and binned into glove pixels by
    their position in that 2D layout. The finger patch rows are split evenly over the three Wuji
    phalanges (2 rows each), the palm grid is spread over the Wuji palm. Several points per pixel; the
    simulated pixel is the strongest of its points (DexTaG's TactileVisualizer).

anatomical  (Benchmark-Genesis, rl_recon/map_tactile_to_wuji.py)
    Each cell is first anchored on the *human* hand, then carried over in coordinates both hands share:
    a finger cell sits at the fraction u of the human finger's length, which falls on one human phalanx
    at a local fraction s; the taxel goes on the same Wuji phalanx at the same s, at the cell's angle
    around the phalanx measured from the palmar direction. A palm cell is placed on the human palm and
    carried by its affine coordinates on (wrist, index MCP, little MCP). One taxel per cell, ray-cast
    onto the convex hulls Genesis collides (a taxel left under a hull could never touch anything).
    The human hand proportions come from the dataset (median over the prepared episodes).
"""

from dataclasses import dataclass

import numpy as np
import trimesh

from tv2s_data.glove import GloveLayout
from tv2s_env.robots.wuji_model import FINGERS, WujiHand

GLOVE_TO_WUJI_FINGER = {"thumb": "thumb", "index": "index", "middle": "middle", "ring": "ring", "pinky": "little"}
PHALANGES = ("link2", "link3", "link4")
NEXT_PART = {"link2": "link3", "link3": "link4", "link4": "tip_link"}
FLEX_JOINT = {"link2": "joint1", "link3": "joint3", "link4": "joint4"}
# MediaPipe chains (base joint .. tip): thumb CMC-MCP-IP-tip, fingers MCP-PIP-DIP-tip
HUMAN_CHAINS = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
                "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}


@dataclass
class TaxelSet:
    side: str
    method: str
    link: np.ndarray        # (P,) Wuji link names
    pos: np.ndarray         # (P, 3) link frame
    normal: np.ndarray      # (P, 3) link frame, outward
    channel: np.ndarray     # (P,) index into the glove layout's assigned channels
    n_channels: int

    def save(self, path: str) -> None:
        np.savez(path, side=self.side, method=self.method, link=self.link, pos=self.pos, normal=self.normal,
                 channel=self.channel, n_channels=self.n_channels)

    @staticmethod
    def load(path: str) -> "TaxelSet":
        d = np.load(path, allow_pickle=False)
        return TaxelSet(side=str(d["side"]), method=str(d["method"]), link=d["link"].astype(str), pos=d["pos"],
                        normal=d["normal"], channel=d["channel"], n_channels=int(d["n_channels"]))


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _perp(v, axis):
    return _unit(v - (v @ axis) * axis)


def _cast(meshes, origin, direction, fallback):
    """First crossing of the ray with any mesh, else the closest surface point to `fallback`."""
    best = None
    for mesh in meshes:
        locs, _, tri = mesh.ray.intersects_location([origin], [direction])
        for loc, f in zip(locs, tri):
            d = np.linalg.norm(loc - origin)
            if best is None or d < best[0]:
                best = (d, loc, mesh.face_normals[f])
    if best is not None:
        return best[1], best[2]
    cands = [trimesh.proximity.closest_point(m, [fallback]) for m in meshes]
    k = int(np.argmin([c[1][0] for c in cands]))
    p, _, t = cands[k]
    return p[0], meshes[k].face_normals[t[0]]


class WujiSurface:
    """The flat Wuji hand (qpos 0) with the directions both mappings measure cell angles from."""

    def __init__(self, hand: WujiHand):
        self.hand = hand
        self.T0 = hand.fk(np.zeros(len(hand.dof_names)))
        pos = lambda l: self.T0[l][:3, 3]
        self.palm = hand.link_name("palm", "")
        self.O = pos(self.palm)
        self.I_mcp, self.L_mcp = pos(hand.link_name("index", "link1")), pos(hand.link_name("little", "link1"))
        n = _unit(np.cross(self.I_mcp - self.O, self.L_mcp - self.O))
        curl = np.mean([self.flexion_direction(f, "link2") for f in FINGERS[1:]], 0)
        self.normal = n * np.sign(n @ curl)                  # palmar
        self.radial = _unit(self.I_mcp - self.L_mcp)         # towards the thumb
        self._hulls: dict[str, list] = {}

    def flexion_direction(self, finger, part):
        q = np.zeros(len(self.hand.dof_names))
        q[self.hand.dof_names.index(f"{self.hand.side}_finger{FINGERS.index(finger) + 1}_{FLEX_JOINT[part]}")] = 0.2
        end = self.hand.link_name(finger, NEXT_PART[part])
        return _unit(self.hand.fk(q)[end][:3, 3] - self.T0[end][:3, 3])

    def palmar(self, finger, part):
        # the thumb metacarpal swings across the palm rather than curling: its palmar side is the palm's
        if finger == "thumb" and part == "link2":
            return self.normal
        return self.flexion_direction(finger, part)

    def hulls(self, link):
        if link not in self._hulls:
            self._hulls[link] = self.hand.collision_hulls(link)
        return self._hulls[link]

    def finger_point(self, finger, part, s, angle):
        """Taxel on phalanx `part` at fraction s along it, `angle` (rad) from palmar towards radial.
        Returns (link, pos, normal) in the link frame."""
        link = self.hand.link_name(finger, part)
        a, b = self.T0[link][:3, 3], self.T0[self.hand.link_name(finger, NEXT_PART[part])][:3, 3]
        u = _unit(b - a)
        p = _perp(self.palmar(finger, part), u)
        r = _perp(self.radial, u)
        r = _unit(r - (r @ p) * p)
        outward = np.cos(angle) * p + np.sin(angle) * r
        return self._cast_link(link, a + s * (b - a), -outward)

    def palm_point(self, a, b):
        """Palm taxel at affine coordinates (a, b) on (O, index MCP, little MCP), cast along -normal."""
        target = self.O + a * (self.I_mcp - self.O) + b * (self.L_mcp - self.O)
        return self._cast_link(self.palm, target, -self.normal)

    def _cast_link(self, link, target_world, direction_world):
        R, t = self.T0[link][:3, :3], self.T0[link][:3, 3]
        tl, dl = R.T @ (target_world - t), R.T @ direction_world
        p, n = _cast(self.hulls(link), tl - 0.04 * dl, dl, tl)
        return link, p, n


def _cell_angle(v, angle_span):
    """Glove column coordinate v (ulnar -> radial) -> angle around the phalanx from the palmar side."""
    return (v - 0.5) * angle_span


def _palm_affine(u, v, wrist, i_mcp, l_mcp, normal, wrist_width_ratio):
    """Palm cell (u: wrist -> knuckles, v: ulnar -> radial) on a palm with these landmarks, as affine
    coordinates (a, b) on (wrist, index MCP, little MCP). The palm is a trapezoid narrowing to the wrist."""
    knuckle = l_mcp + v * (i_mcp - l_mcp)
    width = np.linalg.norm(i_mcp - l_mcp) * wrist_width_ratio
    wrist_pt = wrist + (v - 0.5) * width * _unit(i_mcp - l_mcp)
    p = (1 - u) * wrist_pt + u * knuckle
    A = np.stack([i_mcp - wrist, l_mcp - wrist, normal], 1)
    a, b, _ = np.linalg.solve(A, p - wrist)
    return a, b


def build_layout_taxels(hand: WujiHand, layout: GloveLayout, points_per_cell: int = 2,
                        angle_span: float = np.radians(140), palm_wrist_ratio: float = 0.7) -> TaxelSet:
    """DexTaG-style mapping: dense points on each link, binned into glove pixels by a fixed 2D layout."""
    surf = WujiSurface(hand)
    links, pos, nrm, chan = [], [], [], []
    k = points_per_cell
    for region in ("palm",) + tuple(GLOVE_TO_WUJI_FINGER):
        m = np.flatnonzero(layout.region == region)
        n_r, n_c = len(np.unique(layout.rows[m])), len(np.unique(layout.cols[m]))
        us = (np.arange(n_r * k) + 0.5) / (n_r * k)
        vs = (np.arange(n_c * k) + 0.5) / (n_c * k)
        for u in us:
            for v in vs:
                cell = m[np.argmin((layout.u[m] - u) ** 2 + (layout.v[m] - v) ** 2)]
                if region == "palm":
                    a, b = _palm_affine(u, v, surf.O, surf.I_mcp, surf.L_mcp, surf.normal, palm_wrist_ratio)
                    link, p, n = surf.palm_point(a, b)
                else:
                    finger = GLOVE_TO_WUJI_FINGER[region]
                    seg = min(int(u * 3), 2)                       # 2 glove rows per phalanx
                    link, p, n = surf.finger_point(finger, PHALANGES[seg], u * 3 - seg,
                                                   _cell_angle(v, angle_span))
                links.append(link), pos.append(p), nrm.append(n), chan.append(cell)
    return TaxelSet(side=hand.side, method="layout", link=np.array(links), pos=np.array(pos),
                    normal=np.array(nrm), channel=np.array(chan), n_channels=layout.n_channels)


def human_hand_proportions(keypoints: np.ndarray) -> dict:
    """Median human proportions from (T, 21, 3) MediaPipe keypoints: per-finger phalanx lengths, and the
    palm landmarks in a wrist-centred palm frame (x: little -> index MCP, z: palm normal)."""
    out = {"phalanges": {}}
    for f, chain in HUMAN_CHAINS.items():
        seg = np.linalg.norm(np.diff(keypoints[:, chain], axis=1), axis=-1)       # (T, 3)
        out["phalanges"][f] = np.median(seg, 0)
    w, i, l = keypoints[:, 0], keypoints[:, 5], keypoints[:, 17]
    x = _unit(i - l)
    z = _unit(np.cross(i - w, l - w))
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1)                                                     # (T, 3, 3) columns
    local = lambda p: np.einsum("tji,tj->ti", R, p - w)
    out["palm"] = {"wrist": np.zeros(3), "index_mcp": np.median(local(i), 0), "little_mcp": np.median(local(l), 0)}
    return out


def build_anatomical_taxels(hand: WujiHand, layout: GloveLayout, human: dict,
                            angle_span: float = np.radians(140), palm_wrist_ratio: float = 0.7) -> TaxelSet:
    """Benchmark-Genesis-style mapping: one taxel per cell, through the human hand's proportions."""
    surf = WujiSurface(hand)
    hp = human["palm"]
    h_normal = np.array([0.0, 0.0, 1.0])
    links, pos, nrm, chan = [], [], [], []
    for c in range(layout.n_channels):
        region, u, v = layout.region[c], layout.u[c], layout.v[c]
        if region == "palm":
            a, b = _palm_affine(u, v, hp["wrist"], hp["index_mcp"], hp["little_mcp"], h_normal, palm_wrist_ratio)
            link, p, n = surf.palm_point(a, b)
        else:
            lengths = human["phalanges"][region]
            arc = u * lengths.sum()
            seg = int(min(np.searchsorted(np.cumsum(lengths), arc, side="right"), 2))
            s = (arc - np.r_[0, np.cumsum(lengths)][seg]) / lengths[seg]
            link, p, n = surf.finger_point(GLOVE_TO_WUJI_FINGER[region], PHALANGES[seg], float(np.clip(s, 0, 1)),
                                           _cell_angle(v, angle_span))
        links.append(link), pos.append(p), nrm.append(n), chan.append(c)
    return TaxelSet(side=hand.side, method="anatomical", link=np.array(links), pos=np.array(pos),
                    normal=np.array(nrm), channel=np.array(chan), n_channels=layout.n_channels)
