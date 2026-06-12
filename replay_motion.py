"""
Kinematic replay of a tracking npz on the T1_23dof model — no physics, no policy.
Pure visual check that the motion file is correct: sets root + joints each frame
from body_pos_w[:,0] / body_quat_w[:,0] / joint_pos and renders.

Usage:
    python replay_motion.py output/kick_flavio-004/t1_motion.npz          # save mp4
    python replay_motion.py output/kick_flavio-004/t1_motion.npz --view   # live viewer
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).parent
XML  = ROOT / "models" / "t1_23dof" / "T1_23dof.xml"


def replay(npz_path: str, out_mp4: str | None, live: bool) -> None:
    d = np.load(npz_path, allow_pickle=True)
    body_pos  = d["body_pos_w"]   # (T, nbody, 3) — index 0 is robot root (Trunk)
    body_quat = d["body_quat_w"]  # (T, nbody, 4) wxyz
    joint_pos = d["joint_pos"]    # (T, 23)
    fps = float(d["fps"][0])
    T = joint_pos.shape[0]

    model = mujoco.MjModel.from_xml_path(str(XML))
    data  = mujoco.MjData(model)

    def set_frame(i):
        data.qpos[:3]   = body_pos[i, 0]    # root = first robot body
        data.qpos[3:7]  = body_quat[i, 0]
        data.qpos[7:30] = joint_pos[i]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    if live:
        import time
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as v:
            i = 0
            while v.is_running():
                set_frame(i % T)
                v.sync()
                time.sleep(1.0 / fps)
                i += 1
        return

    import imageio.v2 as imageio
    W, H = 960, 720
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 4.0
    cam.elevation = -15
    cam.azimuth = 130

    frames = []
    for i in range(T):
        set_frame(i)
        cam.lookat[:] = body_pos[i, 0]
        renderer.update_scene(data, cam)
        frames.append(renderer.render())

    out = out_mp4 or str(Path(npz_path).with_name("t1_replay.mp4"))
    imageio.mimsave(out, frames, fps=int(fps))
    print(f"Saved {out}  ({T} frames @ {fps:.0f} fps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--view", action="store_true", help="live viewer instead of mp4")
    args = ap.parse_args()
    replay(args.npz, args.out, args.view)
