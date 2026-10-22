"""Wuji Hand kinematics without pinocchio (vendored from Benchmark-Genesis reconstruction/rl_recon/wuji_model.py): URDF parsing, numpy forward kinematics, link
meshes, and the floating-base wrapper URDF the RL environment drives.

The Wuji Hand (wuji-retargeting's wuji-description) has 5 fingers x 4 revolute joints:
finger1 = thumb, finger2 = index, finger3 = middle, finger4 = ring, finger5 = little.
For finger 2-5, joint1 flexes the MCP, joint2 abducts it, joint3 is the PIP, joint4 the
DIP; link2 is the proximal phalanx, link3 the middle one, link4 the distal one, and
tip_link (fixed) the fingertip. For the thumb, joint1/joint2 form the CMC, link2 is the
metacarpal, link3 the proximal and link4 the distal phalanx. The retargeter places the
palm_link frame on the human wrist (hawor_to_wuji_qpos.py: wrist_pos, wrist_quat).
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_WUJI_DIR = os.environ.get(
    "WUJI_RETARGETING_DIR",
    os.path.expanduser("~/wuji-ego-mint/eval/simulate/wuji-retargeting"),
)

FINGERS = ["thumb", "index", "middle", "ring", "little"]       # finger1 .. finger5
# Keypoints tracked by the hand rewards: PIP, DIP and tip of every finger (Benchmark-Genesis' 15 keypoints)
KEYPOINT_PARTS = ["link3", "link4", "tip_link"]
# The virtual base: three prismatic joints along the world axes, a fixed rotation (the
# trajectory's mean wrist orientation, so the Euler angles stay far from gimbal lock),
# then three revolute joints. palm = R0 Rx(a) Ry(b) Rz(c), at (x, y, z).
BASE_JOINTS = ["base_x", "base_y", "base_z", "base_rx", "base_ry", "base_rz"]


def default_urdf(side, wuji_dir=DEFAULT_WUJI_DIR):
    return os.path.join(wuji_dir, "wuji_retargeting", "wuji-description", "hand", "body", "urdf", f"{side}.urdf")


def rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _origin(element):
    o = element.find("origin") if element is not None else None
    xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
    rpy = np.array([float(v) for v in o.get("rpy", "0 0 0").split()]) if o is not None else np.zeros(3)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rpy_to_matrix(rpy), xyz
    return T


class WujiHand:
    """The hand's kinematic tree, FK for a (20,) or (T, 20) finger configuration."""

    def __init__(self, side="left", urdf=None):
        self.side = side
        self.urdf = os.path.abspath(urdf or default_urdf(side))
        self.root_el = ET.parse(self.urdf).getroot()
        self.links = [l.get("name") for l in self.root_el.findall("link")]
        self.joints = []
        for j in self.root_el.findall("joint"):
            axis = j.find("axis")
            limit = j.find("limit")
            self.joints.append(dict(
                name=j.get("name"), type=j.get("type"),
                parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                origin=_origin(j),
                axis=np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else None,
                lower=float(limit.get("lower")) if limit is not None else 0.0,
                upper=float(limit.get("upper")) if limit is not None else 0.0,
                effort=float(limit.get("effort")) if limit is not None else 0.0,
            ))
        children = {j["child"] for j in self.joints}
        self.root = next(l for l in self.links if l not in children)
        self.dof_names = [j["name"] for j in self.joints if j["type"] == "revolute"]
        self.lower = np.array([j["lower"] for j in self.joints if j["type"] == "revolute"])
        self.upper = np.array([j["upper"] for j in self.joints if j["type"] == "revolute"])
        self.effort = np.array([j["effort"] for j in self.joints if j["type"] == "revolute"])
        # topological order: a joint after its parent link is placed
        order, placed = [], {self.root}
        while len(order) < len(self.joints):
            for j in self.joints:
                if j not in order and j["parent"] in placed:
                    order.append(j)
                    placed.add(j["child"])
        self.order = order

    def link_name(self, finger, part):
        """finger in FINGERS (or 'palm'), part in 'link1'..'link4', 'tip_link'."""
        if finger == "palm":
            return f"{self.side}_palm_link"
        return f"{self.side}_finger{FINGERS.index(finger) + 1}_{part}"

    def fk(self, qpos, base=None):
        """Link poses {name: (..., 4, 4)} for qpos (..., 20) in dof_names order. `base`
        (..., 4, 4) is the palm's pose (identity by default)."""
        qpos = np.asarray(qpos, float)
        batch = qpos.shape[:-1]
        T = {self.root: np.broadcast_to(np.eye(4) if base is None else base, batch + (4, 4)).copy()}
        q = dict(zip(self.dof_names, np.moveaxis(qpos, -1, 0)))
        for j in self.order:
            local = np.broadcast_to(j["origin"], batch + (4, 4)).copy()
            if j["type"] == "revolute":
                a = j["axis"] / np.linalg.norm(j["axis"])
                angle = q[j["name"]]
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                s, c = np.sin(angle)[..., None, None], np.cos(angle)[..., None, None]
                R = np.eye(3) + s * K + (1 - c) * (K @ K)
                M = np.zeros(batch + (4, 4))
                M[..., :3, :3], M[..., 3, 3] = R, 1.0
                local = local @ M
            T[j["child"]] = T[j["parent"]] @ local
        return T

    def collision_mesh(self, link, merge_fixed=True):
        """The link's collision mesh (trimesh) in the link frame. With `merge_fixed`, the
        meshes of links hanging off it by fixed joints are included, as Genesis merges them
        (the fingertip pad is tip_link's mesh, merged into link4)."""
        import trimesh

        el = next(l for l in self.root_el.findall("link") if l.get("name") == link)
        col = el.find("collision")
        parts = []
        if col is not None:
            filename = col.find("geometry").find("mesh").get("filename")
            path = os.path.normpath(os.path.join(os.path.dirname(self.urdf), filename))
            mesh = trimesh.load(path, force="mesh", process=False)
            parts.append(mesh.apply_transform(_origin(col)))
        if merge_fixed:
            for j in self.joints:
                if j["parent"] == link and j["type"] == "fixed":
                    child = self.collision_mesh(j["child"])
                    if child is not None:
                        parts.append(child.apply_transform(j["origin"]))
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)

    def collision_hulls(self, link):
        """The convex hulls Genesis collides for this link (its URDF morph convexifies each mesh
        geom, fixed-joint children merged in), in the link frame: the surface contact happens on."""
        hulls = []
        el = next(l for l in self.root_el.findall("link") if l.get("name") == link)
        if el.find("collision") is not None:
            hulls.append(self.collision_mesh(link, merge_fixed=False).convex_hull)
        for j in self.joints:
            if j["parent"] == link and j["type"] == "fixed":
                hulls += [h.copy().apply_transform(j["origin"]) for h in self.collision_hulls(j["child"])]
        return hulls

    def movable_links(self):
        """Links Genesis keeps (fixed-joint children are merged into their parent)."""
        fixed = {j["child"] for j in self.joints if j["type"] == "fixed"}
        return [l for l in self.links if l not in fixed]

    def joint_origin(self, child_link):
        return next(j for j in self.joints if j["child"] == child_link)

    def write_floating_urdf(self, out_path, base_rotation, base_mass=0.05):
        """The hand on a 6-DoF virtual base (see BASE_JOINTS), mesh paths made absolute.

        Genesis drives every joint of this chain with PD position control, which a
        free joint does not offer for its rotation. `base_rotation` (3, 3) is the fixed
        world rotation R0 between the prismatic and the revolute joints.
        """
        root = ET.fromstring(ET.tostring(self.root_el))
        for mesh in root.iter("mesh"):
            mesh.set("filename", os.path.normpath(os.path.join(os.path.dirname(self.urdf), mesh.get("filename"))))
        names = ["base_world", "base_px", "base_py", "base_pz", "base_rot0", "base_rrx", "base_rry"]
        inertia = 'ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"'
        new = [ET.fromstring(f'<link name="{n}"><inertial><origin xyz="0 0 0" rpy="0 0 0"/>'
                             f'<mass value="{base_mass}"/><inertia {inertia}/></inertial></link>') for n in names]
        from scipy.spatial.transform import Rotation

        r, p, y = Rotation.from_matrix(base_rotation).as_euler("xyz")   # extrinsic xyz == URDF rpy
        chain = [
            ("base_x", "prismatic", "base_world", "base_px", "0 0 0", "1 0 0"),
            ("base_y", "prismatic", "base_px", "base_py", "0 0 0", "0 1 0"),
            ("base_z", "prismatic", "base_py", "base_pz", "0 0 0", "0 0 1"),
            ("base_r0", "fixed", "base_pz", "base_rot0", f"{r} {p} {y}", None),
            ("base_rx", "revolute", "base_rot0", "base_rrx", "0 0 0", "1 0 0"),
            ("base_ry", "revolute", "base_rrx", "base_rry", "0 0 0", "0 1 0"),
            ("base_rz", "revolute", "base_rry", self.root, "0 0 0", "0 0 1"),
        ]
        joints = []
        for name, kind, parent, child, rpy, axis in chain:
            j = ET.Element("joint", name=name, type=kind)
            ET.SubElement(j, "origin", xyz="0 0 0", rpy=rpy)
            ET.SubElement(j, "parent", link=parent)
            ET.SubElement(j, "child", link=child)
            if axis:
                ET.SubElement(j, "axis", xyz=axis)
                lim = "5" if kind == "prismatic" else "6.2832"
                ET.SubElement(j, "limit", lower=f"-{lim}", upper=lim, effort="1000", velocity="100")
            joints.append(j)
        for i, el in enumerate(new + joints):
            root.insert(i, el)
        ET.ElementTree(root).write(out_path)
        return out_path


def base_coordinates(R0, positions, rotations):
    """Virtual-base joint values (T, 6) for palm poses: x, y, z, then XYZ Euler angles of
    R0^T R (intrinsic, the order the revolute chain applies them)."""
    from scipy.spatial.transform import Rotation

    euler = Rotation.from_matrix(np.einsum("ji,tjk->tik", R0, rotations)).as_euler("XYZ")
    return np.concatenate([positions, np.unwrap(euler, axis=0)], axis=1)


def mean_rotation(rotations):
    """Chordal L2 mean of (T, 3, 3) rotations."""
    U, _, Vt = np.linalg.svd(rotations.sum(0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R
