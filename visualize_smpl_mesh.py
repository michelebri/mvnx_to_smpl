"""
SMPL mesh video using pyrender + trimesh (no pytorch3d needed).
Static global camera, Phong shading, same visual style as GVHMR.

Usage:
    python visualize_smpl_mesh.py output/kick_flavio-004/smpl.npz
    python visualize_smpl_mesh.py output/kick_flavio-004/smpl.npz -o output/kick_flavio-004/smpl_mesh.mp4
"""

import os, sys, types, argparse
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")      # headless rendering via NVIDIA EGL

from pathlib import Path
import numpy as np
import cv2

# ── chumpy stub ───────────────────────────────────────────────────────────────
_ch = types.ModuleType("chumpy")
class _Ch:
    def __init__(self, *a, **k): self._r = None
    def __setstate__(self, s):
        if isinstance(s, dict):
            for v in s.values():
                if isinstance(v, np.ndarray) and v.dtype.kind == 'f':
                    self._r = v.astype(np.float32); break
    @property
    def r(self): return self._r if self._r is not None else np.array([])
    def __array__(self, d=None): return self.r
    @property
    def shape(self): return self.r.shape
    def __len__(self): return len(self.r)
_ch.Ch = _Ch
sys.modules.setdefault("chumpy", _ch)
sys.modules.setdefault("chumpy.ch", _ch)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import torch
import articulate as art

import pyrender, trimesh

SMPL_FILE = str(ROOT / "models" / "smpl" / "SMPL_NEUTRAL.pkl")
W, H   = 960, 720
FPS    = 30
COLOR  = [0.65, 0.74, 0.86, 1.0]   # RGBA — soft blue like GVHMR


def _static_camera(verts_all: np.ndarray):
    """Return (4x4 camera pose, yfov) that covers the full trajectory."""
    # centroid of all vertex centroids, XZ plane
    centroids = verts_all.mean(axis=1)          # (T, 3)
    cx = centroids[:, 0].mean()
    cy = 1.0                                     # look at hip height
    cz = centroids[:, 2].mean()

    # radius: max XZ extent from centroid
    dx = centroids[:, 0] - cx
    dz = centroids[:, 2] - cz
    radius = max(np.sqrt(dx**2 + dz**2).max() * 3.5, 2.5)

    # place camera 45° around Y, looking at centroid
    angle = np.radians(45)
    cam_x = cx + radius * np.sin(angle)
    cam_y = cy + radius * np.tan(np.radians(25))
    cam_z = cz + radius * np.cos(angle)

    # look-at matrix
    forward = np.array([cx - cam_x, cy - cam_y, cz - cam_z])
    forward /= np.linalg.norm(forward)
    right = np.cross([0, 1, 0], forward)
    right_n = np.linalg.norm(right)
    if right_n < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= right_n
    up = np.cross(forward, right)

    # pyrender camera looks down -Z by default; flip forward & up
    R = np.stack([right, up, -forward], axis=1)   # (3,3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [cam_x, cam_y, cam_z]
    return T, np.radians(45)


def run(npz_path: str, out_path: str):
    device = torch.device("cpu")

    d = np.load(npz_path)
    local_poses = torch.tensor(d['local_poses'], dtype=torch.float32)   # (T,24,3,3)
    root_pos    = torch.tensor(d['root_positions'], dtype=torch.float32) # (T,3)
    fps = float(d.get('fps', FPS))
    T = local_poses.shape[0]

    # FK → mesh
    body_model = art.model.ParametricModel(SMPL_FILE, device=device)
    print(f"Running FK + mesh for {T} frames…")
    _, _, verts_t = body_model.forward_kinematics(local_poses, tran=root_pos, calc_mesh=True)
    verts_np = verts_t.detach().cpu().numpy()        # (T, V, 3)
    faces_np = body_model.face.astype(np.int32)       # (F, 3)

    cam_pose, yfov = _static_camera(verts_np)

    # pyrender scene (static lights + camera, swap mesh each frame)
    scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0], ambient_light=[0.3, 0.3, 0.3])

    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=W/H)
    scene.add(cam, pose=cam_pose)

    light_pose = cam_pose.copy()
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=light_pose)

    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=COLOR, metallicFactor=0.0, roughnessFactor=0.6)

    r = pyrender.OffscreenRenderer(W, H)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    mesh_node = None
    print(f"Rendering {T} frames…")
    for t in range(T):
        tri = trimesh.Trimesh(vertices=verts_np[t], faces=faces_np, process=False)
        mesh = pyrender.Mesh.from_trimesh(tri, material=mat, smooth=True)

        if mesh_node is not None:
            scene.remove_node(mesh_node)
        mesh_node = scene.add(mesh)

        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        img = color[..., :3]                        # (H, W, 3) RGB uint8
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if t % 30 == 0:
            print(f"  {t}/{T}")

    writer.release()
    r.delete()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('-o', '--out', default=None)
    args = ap.parse_args()
    out = args.out or str(Path(args.npz).parent / "smpl_vis.mp4")
    run(args.npz, out)
