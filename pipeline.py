"""
MVNX -> Booster T1 pipeline.

Usage:
    python pipeline.py input/kick.mvnx                              # single file
    python pipeline.py input/                                        # whole folder

Output for each file <name>.mvnx:
    output/<name>/skeleton.mp4    step 1 - raw MVNX skeleton
    output/<name>/smpl.npz        step 2 - SMPL local_poses
    output/<name>/smpl_vis.mp4    step 3 - SMPL skeleton sanity check
    output/<name>/smplx.npz       step 4 - SMPL-X
    output/<name>/t1.mp4          step 5 - Booster T1 render
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np

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

ROOT        = Path(__file__).parent
SMPL_FILE   = ROOT / "models" / "smpl" / "SMPL_NEUTRAL.pkl"
GMR_DIR     = ROOT / "GMR"
ART_PATH    = ROOT

sys.path.insert(0, str(ART_PATH))
sys.path.insert(0, str(GMR_DIR))

import torch
from scipy.spatial.transform import Rotation as R, Slerp

import articulate as art
import mvnx_to_smpl        as _m2s
import smpl_to_smplx_npz   as _s2sx
import visualize_mvnx       as _vmvnx


def stem(mvnx_path: Path) -> str:
    name = mvnx_path.stem
    name = name.split("#")[0]
    return name.strip().replace(" ", "_")


def run_one(mvnx_path: Path, args):
    name   = stem(mvnx_path)
    outdir = ROOT / "output" / name
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {mvnx_path.name}  →  output/{name}/")
    print(f"{'='*60}")

    if not args.skip_vis:
        out1 = outdir / "skeleton.mp4"
        print(f"[1/5] skeleton video → {out1.name}")
        _vmvnx.make_video(str(mvnx_path), str(out1), target_fps=30)

    out2 = outdir / "smpl.npz"
    print(f"[2/5] MVNX -> SMPL   → {out2.name}")
    _m2s.convert(str(mvnx_path), str(out2))

    out3 = outdir / "smplx.npz"
    print(f"[3/4] SMPL -> SMPL-X → {out3.name}")
    _s2sx.convert(str(out2), str(out3))

    out4     = outdir / "t1.mp4"
    out4_pkl = outdir / "t1_gmr.pkl"
    print(f"[4/5] Booster T1     → {out4.name}")
    _run_render(str(out3), str(out4), str(out4_pkl), args)

    out5 = outdir / "t1_motion.npz"
    print(f"[5/5] tracking npz   → {out5.name}")
    import gmr_to_tracking as _g2t
    _g2t.convert(str(out4_pkl), str(out5))

    print(f"\n  ✓  output/{name}/")
    return outdir


def _run_render(smplx_npz: str, out_mp4: str, out_pkl: str, args):
    """Inline version of render_t1.main() to avoid re-importing on each call."""
    import pathlib, numpy as np
    from scipy.spatial.transform import Rotation as ScipyR
    from general_motion_retargeting import GeneralMotionRetargeting as GMR, RobotMotionViewer
    from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

    smplx_body_models = GMR_DIR / "assets" / "body_models"
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        smplx_npz, smplx_body_models)
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30)

    yup2zup = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    yup2zup_q = ScipyR.from_matrix(yup2zup).as_quat(scalar_first=True)
    for frame in frames:
        for jname in frame:
            pos, quat = frame[jname]
            new_pos  = pos @ yup2zup.T
            new_quat = (ScipyR.from_quat(yup2zup_q, scalar_first=True)
                        * ScipyR.from_quat(quat,     scalar_first=True))
            frame[jname] = (new_pos, new_quat.as_quat(scalar_first=True))

    retarget = GMR(actual_human_height=human_height,
                   src_human="smplx", tgt_robot="booster_t1")
    viewer   = RobotMotionViewer(robot_type="booster_t1", motion_fps=aligned_fps,
                                 record_video=True, video_path=out_mp4,
                                 camera_follow=False)

    qpos_list = [retarget.retarget(frames[i], offset_to_ground=True)
                 for i in range(len(frames))]

    root_xy = np.array([q[:3] for q in qpos_list])
    center  = root_xy.mean(axis=0); center[2] = 0.8
    cam = viewer.viewer.cam if viewer.viewer is not None else None
    if cam is not None:
        cam.lookat[:]  = center
        cam.distance   = max(4.0, float(np.ptp(root_xy[:,0])) + 2.0)
        cam.azimuth    = 140.0
        cam.elevation  = -15.0

    for q in qpos_list:
        viewer.step(root_pos=q[:3], root_rot=q[3:7], dof_pos=q[7:],
                    human_motion_data=None, follow_camera=False, rate_limit=False)
    viewer.close()

    import pickle
    gmr_data = {
        'fps':      int(round(aligned_fps)),
        'root_pos': np.array([q[:3]  for q in qpos_list], dtype=np.float32),
        'root_rot': np.array([q[3:7] for q in qpos_list], dtype=np.float32),
        'dof_pos':  np.array([q[7:]  for q in qpos_list], dtype=np.float32),
    }
    with open(out_pkl, 'wb') as f:
        pickle.dump(gmr_data, f)


def collect_mvnx(path: Path):
    if path.is_file():
        if path.suffix.lower() == ".mvnx":
            return [path]
        sys.exit(f"Error: {path} is not a .mvnx file")
    if path.is_dir():
        files = sorted(path.glob("*.mvnx"))
        if not files:
            sys.exit(f"Error: no .mvnx files found in {path}")
        return files
    sys.exit(f"Error: {path} does not exist")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="MVNX -> Booster T1 pipeline (single file or folder)")
    ap.add_argument("input", help=".mvnx file or folder of .mvnx files")
    ap.add_argument("--skip-vis", action="store_true",
                    help="skip sanity-check skeleton videos (steps 1 and 3), faster")
    args = ap.parse_args()

    files = collect_mvnx(Path(args.input))
    print(f"Found {len(files)} file(s) to process.")

    for f in files:
        run_one(f, args)

    print("\nAll done.")
