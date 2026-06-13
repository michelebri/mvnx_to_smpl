"""
MVNX -> SMPL local_poses .npz

Usage:
    python mvnx_to_smpl.py input/kick.mvnx -o output/smpl.npz
"""

import sys
import types
import numpy as np

_ch_mod = types.ModuleType("chumpy")

class _Ch:
    def __init__(self, *args, **kwargs):
        self.r = None
    def __setstate__(self, state):
        if isinstance(state, dict):
            x = state.get('x', state.get('_x', None))
            if x is not None:
                self.r = np.asarray(x, dtype=np.float32)
            for k, v in state.items():
                if isinstance(v, np.ndarray) and v.dtype.kind == 'f':
                    self.r = v.astype(np.float32)
                    break
    def __array__(self, dtype=None):
        return self.r if self.r is not None else np.array([])

_ch_mod.Ch = _Ch
sys.modules.setdefault("chumpy", _ch_mod)
sys.modules.setdefault("chumpy.ch", _ch_mod)

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp

_ART_PATH = Path(__file__).parent
if str(_ART_PATH) not in sys.path:
    sys.path.insert(0, str(_ART_PATH))

import articulate as art

XSENS_TO_SMPL = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18,
                  5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]


def _strip_ns(tag):
    return tag.split("}")[-1]


def parse_mvnx(path):
    """Parse MVNX -> (orientations, root_positions, src_fps).
    orientations: (T, 23, 4) wxyz global quaternions
    root_positions: (T, 3) pelvis position
    """
    tree = ET.parse(path)
    root = tree.getroot()

    src_fps = None
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag == "subject":
            fr = elem.get("frameRate")
            if fr:
                src_fps = float(fr)
            break

    orientations, positions = [], []
    for elem in root.iter():
        if _strip_ns(elem.tag) != "frame":
            continue
        if elem.get("type") != "normal":
            continue
        quat_el = pos_el = None
        for c in elem:
            t = _strip_ns(c.tag)
            if t == "orientation":
                quat_el = c
            elif t == "position":
                pos_el = c
        if quat_el is None or pos_el is None or not quat_el.text or not pos_el.text:
            continue
        q = np.array(quat_el.text.split(), dtype=np.float32).reshape(23, 4)
        p = np.array(pos_el.text.split(), dtype=np.float32).reshape(23, 3)
        orientations.append(q)
        positions.append(p[0])

    orientations = np.stack(orientations)
    positions    = np.stack(positions)
    return orientations, positions, src_fps or 5.45455


def convert_coord(orientations, root_positions):
    """Xsens frame -> SMPL frame.
    Position: [x,y,z] -> [y,z,x]  (Xsens Y-left,Z-up -> SMPL Y-up,Z-back)
    Quaternion wxyz: [w,x,y,z] -> [w,z,x,y]
    """
    rp = root_positions[:, [1, 2, 0]].copy()

    ori = orientations.copy()
    ori[:, :, 1] = orientations[:, :, 2]
    ori[:, :, 2] = orientations[:, :, 3]
    ori[:, :, 3] = orientations[:, :, 1]
    return ori, rp


def quat_to_rotmat(q):
    """(N,4) wxyz -> (N,3,3)"""
    q = torch.tensor(q, dtype=torch.float32)
    return art.math.quaternion_to_rotation_matrix(q.view(-1, 4)).view(*q.shape[:-1], 3, 3)


def to_smpl_global(orientations):
    """orientations (T,23,4) -> global_poses (T,24,3,3)"""
    T = orientations.shape[0]
    rot = quat_to_rotmat(orientations)
    glb = torch.eye(3).repeat(T, 24, 1, 1)
    for smpl_idx, xsens_idx in enumerate(XSENS_TO_SMPL):
        glb[:, smpl_idx] = rot[:, xsens_idx]
    return glb


def slerp_upsample(local_poses, root_positions, src_fps, tgt_fps):
    """Upsample (T,24,3,3) + (T,3) from src_fps to tgt_fps via slerp."""
    T = local_poses.shape[0]
    if tgt_fps <= src_fps or T < 2:
        return local_poses, root_positions, src_fps

    duration = (T - 1) / src_fps
    src_t = np.arange(T) / src_fps
    M = int(round(duration * tgt_fps)) + 1
    dst_t = np.linspace(0, duration, M)

    rp = np.stack([np.interp(dst_t, src_t, root_positions[:, k]) for k in range(3)], axis=-1)

    lp_np = local_poses.numpy()
    from scipy.spatial.transform import Rotation as R, Slerp
    out = np.zeros((M, 24, 3, 3), dtype=np.float32)
    for j in range(24):
        rots = R.from_matrix(lp_np[:, j])
        sl = Slerp(src_t, rots)
        out[:, j] = sl(dst_t).as_matrix()

    return torch.from_numpy(out), rp, tgt_fps


def convert(mvnx_path, out_path, target_fps=30.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_file = Path(__file__).parent / "models" / "smpl" / "SMPL_NEUTRAL.pkl"
    body_model = art.model.ParametricModel(str(smpl_file), device=device)

    ori, rp, src_fps = parse_mvnx(mvnx_path)
    print(f"  Parsed {len(ori)} frames @ {src_fps:.2f} Hz")

    ori, rp = convert_coord(ori, rp)
    global_poses = to_smpl_global(ori).to(device)
    local_poses  = body_model.inverse_kinematics_R(global_poses)
    local_poses  = local_poses.view(global_poses.shape[0], 24, 3, 3).cpu()
    global_poses = global_poses.cpu()

    local_poses, rp, out_fps = slerp_upsample(local_poses, rp, src_fps, target_fps)

    T = local_poses.shape[0]
    np.savez_compressed(
        out_path,
        global_poses=global_poses.numpy().astype(np.float32),
        local_poses=local_poses.numpy().astype(np.float32),
        root_positions=rp.astype(np.float32),
        fps=np.array(out_fps, dtype=np.float32),
    )
    print(f"  Saved {out_path}  ({T} frames @ {out_fps:.1f} fps)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mvnx")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--target-fps", type=float, default=30.0)
    args = ap.parse_args()
    out = args.out or str(Path(args.mvnx).with_suffix("_smpl.npz"))
    convert(args.mvnx, out, target_fps=args.target_fps)
