"""
Render an SMPL-X .npz onto the Booster T1 robot (fixed camera).

Usage:
    python render_t1.py output/smplx.npz -o output/t1.mp4
"""

import argparse
import pathlib
import sys

import numpy as np
import mujoco as mj

GMR_DIR = pathlib.Path(__file__).parent / "GMR"
sys.path.insert(0, str(GMR_DIR))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
)
from scipy.spatial.transform import Rotation as ScipyR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("smplx_file")
    ap.add_argument("-o", "--out", default="output/t1_render.mp4")
    ap.add_argument("--robot", default="booster_t1")
    ap.add_argument("--tgt-fps", type=int, default=30)
    ap.add_argument("--cam-distance", type=float, default=4.0,
                    help="fixed camera distance")
    ap.add_argument("--cam-azimuth", type=float, default=140.0)
    ap.add_argument("--cam-elevation", type=float, default=-15.0)
    ap.add_argument("--follow", action="store_true",
                    help="follow the robot instead of a fixed camera")
    args = ap.parse_args()

    smplx_body_models = GMR_DIR / "assets" / "body_models"
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        args.smplx_file, smplx_body_models
    )
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.tgt_fps
    )

    yup_to_zup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    yup_to_zup_quat = ScipyR.from_matrix(yup_to_zup).as_quat(scalar_first=True)
    for frame in frames:
        for joint_name in frame.keys():
            pos, quat = frame[joint_name]
            new_pos = pos @ yup_to_zup.T
            new_quat_r = ScipyR.from_quat(yup_to_zup_quat, scalar_first=True) * ScipyR.from_quat(quat, scalar_first=True)
            frame[joint_name] = (new_pos, new_quat_r.as_quat(scalar_first=True))

    retarget = GMR(actual_human_height=human_height, src_human="smplx",
                   tgt_robot=args.robot)

    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        record_video=True,
        video_path=args.out,
        camera_follow=args.follow,
    )

    qpos_list = [retarget.retarget(frames[i], offset_to_ground=True)
                 for i in range(len(frames))]
    root_xy = np.array([q[:3] for q in qpos_list])
    center = root_xy.mean(axis=0)
    center[2] = 0.8  # look at torso height

    if not args.follow:
        cam = viewer.viewer.cam if viewer.viewer is not None else None
        if cam is not None:
            cam.lookat[:] = center
            cam.distance = max(args.cam_distance, float(np.ptp(root_xy[:, 0])) + 2.0)
            cam.azimuth = args.cam_azimuth
            cam.elevation = args.cam_elevation

    for q in qpos_list:
        viewer.step(
            root_pos=q[:3], root_rot=q[3:7], dof_pos=q[7:],
            human_motion_data=None,
            follow_camera=args.follow,
            rate_limit=False,
        )
    viewer.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
