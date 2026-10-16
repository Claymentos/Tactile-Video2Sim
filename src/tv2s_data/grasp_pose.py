"""The in-hand pose of a held object that best explains the glove (Benchmark-Genesis' tactile_grasp_pose.py).

Over one hold (frames where the glove reports the grasp), the reference Wuji hand (the retargeted mocap
hand: palm pose and finger joints per frame) is taken as given and ONE object pose in the palm frame is
optimised -- Digital-Tactile-Twin's optimize_object_pose.py loss on the Wuji taxels (the glove cells carried
onto the Wuji by the tactile mapping):
  glove contact     softplus((d - r) / tau)   pulls the channel's taxels within the probe radius r
  glove no contact  softplus((r - d) / tau)   pushes them out (weight w_neg)
  hand penetration  relu(-sdf) of points sampled on the hand's collision hulls (weight w_pen)
  table             relu(-z) of the object's vertices in the world, every frame of the hold (weight w_table)
  video             relu(|offset - video offset| - margin)^2 (weight w_video): a weak pull towards the pose
                    the camera saw, which monocular depth only places within ~10 cm
  support           (lowest point - floor)^2 at the hold's first / last frame when the object is picked up
                    from / put down on a known support (Gemini's clip: the table, a container's bottom), weight
                    w_support: the grasp must reach the object where it can rest
d is the signed distance to the object's convex hull (the collision shape Genesis simulates: max over the
hull's face planes, negative inside). Starts: the contact taxels' centroid, the video orientation perturbed
(a quarter fully random); the lowest loss is kept.

Unlike Benchmark-Genesis, whose object track came from a sensor-quality video, here the hands (mocap) are
the trusted part: the object is placed in them, the video only breaks ties.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., :3], x[..., 3:]
    e1 = F.normalize(a, dim=-1)
    e2 = F.normalize(b - (e1 * b).sum(-1, keepdim=True) * e1, dim=-1)
    return torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], -1)


@dataclass
class GraspFit:
    offset: np.ndarray          # (4, 4) object pose in the palm frame
    f1: float                   # glove contact F1 of the simulated taxels at that pose
    precision: float
    recall: float
    penetration_mm: float       # mean hand penetration
    video_cm: float             # distance to the video offset


def fit_grasp(hand, taxels, qpos: np.ndarray, palm: np.ndarray, contact: np.ndarray, vertices: np.ndarray,
              video_offset: np.ndarray, radius: float = 0.004, tau: float = 0.0015, w_neg: float = 0.5,
              w_pen: float = 20.0, w_table: float = 20.0, w_video: float = 0.2, video_margin: float = 0.05,
              starts: int = 64, iters: int = 400, max_frames: int = 15, device: str = "cuda", seed: int = 0,
              floors: tuple = (None, None), w_support: float = 2.0) -> GraspFit:
    """hand: WujiHand; taxels: TaxelSet; qpos (F, 20), palm (F, 4, 4) world, contact (F, C) bool glove contacts,
    vertices (V, 3) object mesh (object frame), video_offset (4, 4) the video's object pose in the palm frame,
    floors (first, last): the support height under the object at the hold's first / last frame, or None."""
    import trimesh

    ends = palm[[0, -1]]
    keep = np.linspace(0, len(qpos) - 1, min(max_frames, len(qpos))).round().astype(int)
    qpos, palm, contact = qpos[keep], palm[keep], contact[keep]
    T = hand.fk(qpos)
    P = np.stack([np.einsum("fij,j->fi", T[str(l)][:, :3, :3], p) + T[str(l)][:, :3, 3]
                  for l, p in zip(taxels.link, taxels.pos)], 1)                          # (F, N, 3) palm frame
    surf = []
    for link in hand.movable_links():
        hulls = hand.collision_hulls(link)
        if hulls:
            pts = np.concatenate([h.sample(30) for h in hulls])
            surf.append(np.einsum("fij,nj->fni", T[link][:, :3, :3], pts) + T[link][:, None, :3, 3])
    S = np.concatenate(surf, 1)[:, ::3]
    sub = np.random.default_rng(seed).choice(len(vertices), min(len(vertices), 1000), replace=False)
    hull = trimesh.Trimesh(vertices[sub]).convex_hull                                  # ~500 planes
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    normals, d0 = t(hull.face_normals), t((hull.face_normals * hull.vertices[hull.faces[:, 0]]).sum(1))
    sdf = lambda X: (X @ normals.T - d0).amax(-1)                                       # convex: max over planes
    P, S, y = t(P), t(S), t(contact.astype(np.float32))
    channel = torch.as_tensor(taxels.channel, device=device).long()
    C = y.shape[1]
    Rw, pw = t(palm[:, :3, :3]), t(palm[:, :3, 3])
    Re, pe = t(ends[:, :3, :3]), t(ends[:, :3, 3])
    V = t(vertices[:: max(1, len(vertices) // 400)])

    g = torch.Generator(device=device).manual_seed(seed)
    cell = y[:, channel] > 0
    centroid = (P * cell[..., None]).sum((0, 1)) / cell.sum().clamp(min=1)
    base6 = t(video_offset[:3, :3][:, :2].T.reshape(6))
    noise = 0.3 * torch.randn(starts, 6, generator=g, device=device)
    noise[3 * starts // 4:] = 3.0 * torch.randn(starts - 3 * starts // 4, 6, generator=g, device=device)
    rot6 = (base6[None] + noise).requires_grad_(True)
    trans = (centroid + 0.02 * torch.randn(starts, 3, generator=g, device=device)).requires_grad_(True)
    p_video = t(video_offset[:3, 3])
    opt = torch.optim.Adam([rot6, trans], lr=3e-3)

    def losses():
        R = rot6d_to_rotmat(rot6)                                                         # object -> palm
        to_obj = lambda X: torch.einsum("bji,bfnj->bfni", R, X[None] - trans[:, None, None])
        d_cell = sdf(to_obj(P))                                                           # (B, F, N)
        d = torch.full((starts, len(keep), C), float("inf"), device=device).scatter_reduce(
            2, channel.expand(starts, len(keep), -1), d_cell, reduce="amin")
        pos = (F.softplus((d - radius) / tau) * tau * y).sum((1, 2)) / y.sum().clamp(min=1)
        neg = (F.softplus((radius - d) / tau) * tau * (1 - y)).sum((1, 2)) / (1 - y).sum().clamp(min=1)
        pen = torch.relu(-sdf(to_obj(S))).mean((1, 2))
        world = torch.einsum("fij,bjk,nk->bfni", Rw, R, V) + (pw[None] + torch.einsum("fij,bj->bfi", Rw, trans))[:, :, None]
        table = torch.relu(-world[..., 2]).mean((1, 2))
        video = torch.relu((trans - p_video).norm(dim=-1) - video_margin) ** 2
        support = torch.zeros_like(video)
        for e, floor in enumerate(floors):
            if floor is not None:
                z = torch.einsum("j,bjk,nk->bn", Re[e, 2], R, V) + (pe[e, 2] + trans @ Re[e, 2])[:, None]
                support = support + (-0.005 * torch.logsumexp(-z / 0.005, -1) - floor) ** 2
        return pos + w_neg * neg + w_pen * pen + w_table * table + w_video * video + w_support * support, d, pen

    for _ in range(iters):
        loss = losses()[0]
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        loss, d, pen = losses()
        b = int(torch.argmin(loss))
        sim = d[b] < radius
        tp = (sim & (y > 0)).sum().item()
        prec, rec = tp / max(sim.sum().item(), 1), tp / max(y.sum().item(), 1)
        off = np.eye(4)
        off[:3, :3] = rot6d_to_rotmat(rot6)[b].cpu().numpy()
        off[:3, 3] = trans[b].cpu().numpy()
    return GraspFit(offset=off, f1=2 * prec * rec / max(prec + rec, 1e-9), precision=prec, recall=rec,
                    penetration_mm=float(pen[b]) * 1000, video_cm=float((trans[b] - p_video).norm()) * 100)
