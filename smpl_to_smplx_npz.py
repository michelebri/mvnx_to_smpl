"""
Convert flavio_smpl.npz (SMPL local_poses) -> SMPL-X format NPZ for GMR.

SMPL local_poses (T,24,3,3):
  joint 0  = root (-> root_orient, axis-angle (T,3))
  joints 1-23 = body (-> pose_body, axis-angle (T,69) but SMPL-X uses 21 joints = 63)

SMPL-X pose_body uses joints 1-21 (22 body joints minus root minus 2 hands = 21 joints = 63 vals)
SMPL has 24 joints: root(0), pelvis children (1,2,3), spine (3,6,9), neck(12), head(15),
  shoulders (13,14,16,17)...

Since SMPL and SMPL-X share the first 22 body joints (0-21 in local pose space),
we take:
  root_orient = local_poses[:,0] as axis-angle (T,3)
  pose_body   = local_poses[:,1:22] as axis-angle (T,63)

Usage:
    python smpl_to_smplx_npz.py output/flavio_smpl.npz -o output/flavio_smplx.npz
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


def rotmat_to_axisangle(rotmats):
    """(T, J, 3, 3) -> (T, J, 3)"""
    T, J, _, _ = rotmats.shape
    flat = rotmats.reshape(-1, 3, 3)
    aa = R.from_matrix(flat).as_rotvec()  # (T*J, 3)
    return aa.reshape(T, J, 3).astype(np.float32)


def convert(smpl_npz, out_path):
    d = np.load(smpl_npz)
    local_poses = d['local_poses']   # (T, 24, 3, 3)
    root_pos = d['root_positions']   # (T, 3)
    fps = float(d.get('fps', 30))
    T = local_poses.shape[0]

    aa = rotmat_to_axisangle(local_poses)  # (T, 24, 3)

    root_orient = aa[:, 0, :]        # (T, 3)
    pose_body = aa[:, 1:22, :].reshape(T, 63)  # (T, 63)

    betas = np.zeros(10, dtype=np.float32)
    gender = 'neutral'

    np.savez_compressed(
        out_path,
        pose_body=pose_body,
        root_orient=root_orient,
        trans=root_pos.astype(np.float32),
        betas=betas,
        gender=np.array(gender),
        mocap_frame_rate=np.array(fps, dtype=np.float32),
    )
    print(f"Saved {out_path}  ({T} frames @ {fps:.1f} fps)")
    print(f"  root_orient: {root_orient.shape}")
    print(f"  pose_body:   {pose_body.shape}")
    print(f"  trans:       {root_pos.shape}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('smpl_npz')
    ap.add_argument('-o', '--out', default=None)
    args = ap.parse_args()
    out = args.out or args.smpl_npz.replace('_smpl.npz', '_smplx.npz')
    convert(args.smpl_npz, out)
