"""
Step 3: Visualize SMPL poses as a skeleton video using articulate FK.
Usage:
    python visualize_smpl.py output/flavio_smpl.npz
"""
import sys, types, argparse
import numpy as np
from pathlib import Path

# chumpy stub
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
sys.modules.setdefault("chumpy", _ch); sys.modules.setdefault("chumpy.ch", _ch)

import torch
_ART = Path(__file__).parent
sys.path.insert(0, str(_ART))
import articulate as art

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa

SMPL_FILE = str(Path(__file__).parent / "models" / "smpl" / "SMPL_NEUTRAL.pkl")
PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]


def compute_joints(body_model, local_poses_np, root_positions_np):
    """Run FK and return joint world positions (T,24,3)."""
    lp = torch.tensor(local_poses_np, dtype=torch.float32)   # (T,24,3,3)
    rp = torch.tensor(root_positions_np, dtype=torch.float32) # (T,3)
    # forward_kinematics returns (global_rots (T,24,3,3), joint_positions (T,24,3))
    _, joint_pos = body_model.forward_kinematics(lp, calc_mesh=False)
    # joint_pos is relative to root; add root translation
    joint_pos = joint_pos.cpu().numpy() + root_positions_np[:, None, :]  # (T,24,3)
    return joint_pos


def run(npz_path, out_path, every=1):
    d = np.load(npz_path)
    local_poses  = d['local_poses']    # (T,24,3,3)
    root_pos     = d['root_positions']  # (T,3)
    fps = float(d.get('fps', 30))
    T = local_poses.shape[0]

    body_model = art.model.ParametricModel(SMPL_FILE)
    print(f"Computing FK for {T} frames...")
    all_joints = compute_joints(body_model, local_poses, root_pos)  # (T,24,3)

    # subsample
    all_joints = all_joints[::every]
    Tf = len(all_joints)

    lo = all_joints.reshape(-1,3).min(0)
    hi = all_joints.reshape(-1,3).max(0)
    pad = 0.2

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    lines = [ax.plot([],[],[], 'o-', color='steelblue', lw=2, ms=3)[0] for _ in range(24)]
    ax.set_xlim(lo[0]-pad, hi[0]+pad)
    ax.set_ylim(lo[1]-pad, hi[1]+pad)
    ax.set_zlim(lo[2]-pad, hi[2]+pad)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=15, azim=45)
    title = ax.set_title('')

    def update(t):
        J = all_joints[t]
        for j, line in enumerate(lines):
            p = PARENTS[j]
            if p < 0:
                line.set_data([J[j,0]], [J[j,1]]); line.set_3d_properties([J[j,2]])
            else:
                line.set_data([J[p,0],J[j,0]], [J[p,1],J[j,1]])
                line.set_3d_properties([J[p,2],J[j,2]])
        title.set_text(f'SMPL frame {t}/{Tf-1}')
        return lines + [title]

    ani = animation.FuncAnimation(fig, update, frames=Tf,
                                  interval=1000/(fps/every), blit=False)
    writer = animation.FFMpegWriter(fps=fps/every, bitrate=2000)
    ani.save(out_path, writer=writer)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('-o','--out', default=None)
    ap.add_argument('--every', type=int, default=1)
    args = ap.parse_args()
    out = args.out or str(Path(args.npz).with_suffix('_smpl_vis.mp4'))
    run(args.npz, out, every=args.every)
