"""
Convert a GMR pkl (root_pos, root_rot wxyz, dof_pos[23]) to a mjlab-compatible
tracking npz using the T1_23dof MuJoCo model from colosseum.

GMR retargets onto T1_serial.xml, whose 23-joint order is IDENTICAL to
colosseum's T1_23dof.xml (verified joint-by-joint), so dof_pos maps 1:1
onto qpos[7:30] — no reordering, no default fill.

Output keys match colosseum's convert_replay_to_tracking.py:
    joint_pos, joint_vel, body_pos_w, body_quat_w,
    body_lin_vel_w, body_ang_vel_w, fps, body_names

Usage:
    python gmr_to_tracking.py output/kick/t1_gmr.pkl output/kick/t1_motion.npz
"""

import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm

ROOT       = Path(__file__).parent
XML        = ROOT / "models" / "t1_23dof" / "T1_23dof.xml"
# colosseum's mimic env steps at decimation(4) * timestep(0.005) = 0.02 s = 50 Hz.
# MotionCommand advances one frame per env step and assumes the clip is sampled
# at that rate, so the tracking npz MUST be 50 Hz (matches convert_replay_to_tracking).
OUTPUT_FPS = 50.0

# Joint order shared by GMR's T1_serial.xml and colosseum's T1_23dof.xml.
JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]


def _all_body_names(model) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]


def _resample(root_pos, root_rot, dof_pos, src_fps, dst_fps):
    """Resample a clip from src_fps to dst_fps (lerp for pos/dof, slerp for root quat)."""
    if abs(src_fps - dst_fps) < 1e-6:
        return root_pos, root_rot, dof_pos

    T = root_pos.shape[0]
    duration = (T - 1) / src_fps
    n_out = int(round(duration * dst_fps)) + 1

    t_src = np.arange(T) / src_fps
    t_dst = np.arange(n_out) / dst_fps
    t_dst = np.clip(t_dst, 0.0, t_src[-1])

    # linear interp for positions and joint angles
    rp = np.stack([np.interp(t_dst, t_src, root_pos[:, k]) for k in range(3)], axis=1)
    dp = np.stack([np.interp(t_dst, t_src, dof_pos[:, k]) for k in range(dof_pos.shape[1])], axis=1)

    # slerp for root orientation (input wxyz -> scipy xyzw)
    key_rots = R.from_quat(root_rot[:, [1, 2, 3, 0]])
    slerp = Slerp(t_src, key_rots)
    rr_xyzw = slerp(t_dst).as_quat()
    rr = rr_xyzw[:, [3, 0, 1, 2]]  # back to wxyz

    return rp, rr, dp


def convert(pkl_path: str, out_path: str, fps: float = OUTPUT_FPS) -> None:
    with open(pkl_path, "rb") as f:
        gmr = pickle.load(f)

    root_pos = np.array(gmr["root_pos"], dtype=np.float64)   # (T, 3)
    root_rot = np.array(gmr["root_rot"], dtype=np.float64)   # (T, 4) wxyz
    dof_pos  = np.array(gmr["dof_pos"],  dtype=np.float64)   # (T, 23)
    src_fps  = float(gmr["fps"])

    assert dof_pos.shape[1] == 23, f"expected 23 dofs, got {dof_pos.shape[1]}"

    # Resample to the env step rate (50 Hz) so MotionCommand plays back at real time.
    root_pos, root_rot, dof_pos = _resample(root_pos, root_rot, dof_pos, src_fps, fps)
    T = root_pos.shape[0]
    if abs(src_fps - fps) > 1e-6:
        print(f"Resampled {src_fps:.0f} Hz -> {fps:.0f} Hz: {T} frames")

    model = mujoco.MjModel.from_xml_path(str(XML))
    data  = mujoco.MjData(model)
    model.opt.timestep = 1.0 / fps

    # qpos addresses of the 23 joints, in the GMR/colosseum shared order.
    qpos_adr = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"joint {name} not in model"
        qpos_adr.append(model.jnt_qposadr[jid])
    qpos_adr = np.array(qpos_adr)
    # Sanity: must be the contiguous 7..29 block.
    assert qpos_adr.tolist() == list(range(7, 30)), qpos_adr.tolist()

    body_names = _all_body_names(model)
    print(f"T1_23dof: {model.nbody} bodies, {len(JOINT_NAMES)} joints")
    print(f"Input: {T} frames, {dof_pos.shape[1]} dofs, src_fps={gmr['fps']}")

    # Ground-offset: GMR retargets onto T1_serial.xml, whose foot geoms sit a few
    # cm higher off the ground than colosseum's T1_23dof feet. Drop the whole
    # trajectory by a constant so the lowest collision geom over the clip rests
    # on z=0 (matches colosseum's spawn where feet touch the floor).
    min_geom_z = np.inf
    for i in range(T):
        data.qpos[:3]   = root_pos[i]
        data.qpos[3:7]  = root_rot[i]
        data.qpos[7:30] = dof_pos[i]
        mujoco.mj_forward(model, data)
        min_geom_z = min(min_geom_z, float(data.geom_xpos[:, 2].min()))
    root_pos = root_pos.copy()
    root_pos[:, 2] -= min_geom_z
    print(f"Ground offset: lowered root by {min_geom_z:.4f} m")

    log: dict[str, list] = {
        "joint_pos": [], "joint_vel": [],
        "body_pos_w": [], "body_quat_w": [],
        "body_lin_vel_w": [], "body_ang_vel_w": [],
    }

    dt = 1.0 / fps
    prev_xpos = prev_xquat = None

    for i in tqdm(range(T)):
        data.qpos[:3]    = root_pos[i]
        data.qpos[3:7]   = root_rot[i]
        data.qpos[7:30]  = dof_pos[i]      # 1:1 mapping
        data.qvel[:]     = 0.0
        mujoco.mj_forward(model, data)

        # Drop MuJoCo body 0 ('world'): mjlab's robot.data.body_link_pos_w (the
        # reference format) indexes robot bodies only, with Trunk at index 0.
        # MotionCommand uses body_pos_w[:, 0] as the robot root, so 'world' here
        # would spawn the robot at the origin with the torso on the floor.
        xpos  = data.xpos[1:].copy()
        xquat = data.xquat[1:].copy()

        if prev_xpos is None:
            linvel = np.zeros_like(xpos)
            angvel = np.zeros_like(xpos)
        else:
            linvel = (xpos - prev_xpos) / dt
            dq = (R.from_quat(xquat[:, [1, 2, 3, 0]])
                  * R.from_quat(prev_xquat[:, [1, 2, 3, 0]]).inv())
            angvel = dq.as_rotvec() / dt

        prev_xpos, prev_xquat = xpos, xquat

        log["joint_pos"].append(data.qpos[7:30].copy().astype(np.float32))
        log["joint_vel"].append(np.zeros(23, dtype=np.float32))  # static IK frame
        log["body_pos_w"].append(xpos.astype(np.float32))
        log["body_quat_w"].append(xquat.astype(np.float32))
        log["body_lin_vel_w"].append(linvel.astype(np.float32))
        log["body_ang_vel_w"].append(angvel.astype(np.float32))

    # joint_vel by finite difference (cleaner than leaving zeros)
    jp = np.stack(log["joint_pos"])
    jv = np.zeros_like(jp)
    jv[1:] = (jp[1:] - jp[:-1]) / dt
    jv[0]  = jv[1]
    log["joint_vel"] = list(jv)

    log["body_lin_vel_w"][0] = log["body_lin_vel_w"][1]
    log["body_ang_vel_w"][0] = log["body_ang_vel_w"][1]

    out = {k: np.stack(v, axis=0) for k, v in log.items()}
    out["fps"]        = np.array([fps], dtype=np.float32)
    out["body_names"] = np.array(body_names[1:])  # drop 'world' to match body_pos_w

    np.savez(out_path, **out)
    print(f"Saved {out_path}")
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=OUTPUT_FPS)
    args = ap.parse_args()
    convert(args.pkl, args.out, args.fps)
