"""Distance from the taxels to reconstructed meshes, for the simulated glove (tactile/sensor.py).

RecGen meshes are open shells (the side the camera saw), so there is no inside to sign a distance with:
the glove reads the unsigned distance to the surface, which is what the taxel model needs
(pressure = kn * max(probe_radius - d, 0) is zero beyond the probe radius either way, and a taxel
pressed into the surface is within it).

Per object, the distance is precomputed once on a voxel grid around the mesh (object frame, --voxel
3 mm, --margin 3 cm), from a k-d tree of points sampled densely on the surface (an error below the
sample spacing, ~1 mm), and cached next to the mesh (sdf_<voxel>mm.npz). At run time a batch of points
is read with trilinear interpolation (torch grid_sample); outside the grid, the distance is the grid
boundary's plus the distance to the grid box, so it keeps growing.

`signed_distance_grid` / `MeshDistance(signed=True)`: Benchmark-Genesis' VoxelSDF (Digital-Tactile-Twin
sim/object_sdf.py) for the simulated geometry (rl_recon_env.py: the decimated collision mesh), negative
inside: trimesh's exact signed distance for meshes up to 5000 faces, else the filled voxel occupancy and a
Euclidean distance transform on each side (~half a voxel accurate; a mesh that is not watertight fills to
its shell only). Cached in outputs/sdf_cache, keyed by the mesh's content.

`collision_mesh`: Benchmark-Genesis' collision geometry (rl_recon/prepare_reference.py: the mesh decimated
to --collision-faces): the mesh itself if it is small enough, else a decimated copy next to it.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def distance_grid(mesh_path: str | Path, voxel: float = 0.003, margin: float = 0.03, samples: int = 200_000):
    """(grid (D, H, W) float32 metres, lower corner (3,), voxel) for a mesh in its own frame, cached."""
    mesh_path = Path(mesh_path)
    cache = mesh_path.parent / f"sdf_{voxel * 1000:g}mm.npz"
    if cache.exists() and cache.stat().st_mtime >= mesh_path.stat().st_mtime:
        d = np.load(cache)
        return d["grid"], d["lower"], float(d["voxel"])
    import trimesh
    from scipy.spatial import cKDTree

    mesh = trimesh.load(mesh_path, force="mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, samples, seed=0)
    lower = mesh.bounds[0] - margin
    upper = mesh.bounds[1] + margin
    n = np.ceil((upper - lower) / voxel).astype(int) + 1
    axes = [lower[k] + voxel * np.arange(n[k]) for k in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")                       # (X, Y, Z)
    q = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    dist, _ = cKDTree(pts).query(q, workers=-1)
    grid = dist.reshape(n).astype(np.float32).transpose(2, 1, 0)        # (Z, Y, X): grid_sample's (D, H, W)
    np.savez_compressed(cache, grid=grid, lower=lower, voxel=voxel)
    return grid, lower, voxel


def collision_mesh(mesh_path: str | Path, faces: int = 4000) -> Path:
    """The mesh decimated to at most `faces` faces (collision_<faces>.obj next to it), cached."""
    mesh_path = Path(mesh_path)
    out = mesh_path.parent / f"collision_{faces}.obj"
    if out.exists() and out.stat().st_mtime >= mesh_path.stat().st_mtime:
        return out
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    if len(mesh.faces) <= faces:
        return mesh_path
    mesh.simplify_quadric_decimation(face_count=faces).export(out)
    return out


def signed_distance_grid(mesh_path: str | Path, voxel: float = 0.001, margin: float = 0.03, max_cells: int = 160):
    """(grid (D, H, W) float32 metres, negative inside, lower corner (3,), voxel), cached. The voxel grows on
    large objects so that no side exceeds `max_cells` (Benchmark-Genesis' VoxelSDF)."""
    import hashlib

    import trimesh

    mesh_path = Path(mesh_path)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    lower, upper = mesh.bounds[0] - margin, mesh.bounds[1] + margin
    voxel = max(voxel, float((upper - lower).max()) / (max_cells - 1))
    key = hashlib.md5(mesh.vertices.tobytes() + mesh.faces.tobytes() + np.array([voxel, margin]).tobytes()).hexdigest()
    cache = Path(__file__).resolve().parents[3] / "outputs" / "sdf_cache" / f"{key}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["grid"], d["lower"], float(d["voxel"])
    n = np.ceil((upper - lower) / voxel).astype(int) + 1
    if len(mesh.faces) <= 5000:
        axes = [lower[i] + voxel * np.arange(n[i]) for i in range(3)]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        grid = -np.concatenate([trimesh.proximity.signed_distance(mesh, c)          # trimesh: + inside
                                for c in np.array_split(P, max(1, len(P) // 50000))]).reshape(*n)
    else:
        from scipy.ndimage import distance_transform_edt

        filled = mesh.voxelized(pitch=voxel).fill()
        idx = np.clip(np.round((filled.points - lower) / voxel).astype(int), 0, n - 1)
        occ = np.zeros(n, bool)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        grid = np.where(occ, -(distance_transform_edt(occ) - 0.5), distance_transform_edt(~occ) - 0.5) * voxel
    grid = grid.astype(np.float32).transpose(2, 1, 0)                     # (Z, Y, X): grid_sample's (D, H, W)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, grid=grid, lower=lower, voxel=voxel)
    return grid, lower, voxel


class MeshDistance:
    """Distance to one mesh for batches of points in its frame: unsigned (`distance_grid`), or signed
    (`signed_distance_grid`, negative inside)."""

    def __init__(self, mesh_path: str | Path, device: torch.device, voxel: float = 0.003, signed: bool = False):
        grid, lower, voxel = (signed_distance_grid if signed else distance_grid)(mesh_path, voxel)
        self.grid = torch.as_tensor(grid, device=device)[None, None]    # (1, 1, D, H, W)
        self.lower = torch.as_tensor(lower, dtype=torch.float32, device=device)
        D, H, W = grid.shape
        self.upper = self.lower + voxel * torch.tensor([W - 1, H - 1, D - 1], dtype=torch.float32, device=device)

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        """p (..., 3) object frame -> (...) distance (m)."""
        shape = p.shape[:-1]
        q = p.reshape(-1, 3)
        inside = torch.minimum(torch.maximum(q, self.lower), self.upper)
        outside = (q - inside).norm(dim=-1)                              # distance to the grid box
        g = 2 * (inside - self.lower) / (self.upper - self.lower) - 1   # [-1, 1], (x, y, z) order
        d = F.grid_sample(self.grid, g[None, :, None, None, :], mode="bilinear", align_corners=True)
        return (d.reshape(-1) + outside).reshape(shape)
