"""
Step 1: Visualize raw MVNX skeleton as a video.
Uses only segment POSITIONS (the clean signal) to draw a 3D stick figure.
No SMPL, no retargeting — just "what is actually in the mocap file".

Usage:
    python visualize_mvnx.py input/kick_flavio-004#MVN\ System\ 2.mvnx
    python visualize_mvnx.py input/kick_flavio-004#MVN\ System\ 2.mvnx --target-fps 30
"""

import argparse
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa


# Xsens 23-segment skeleton connectivity (parent -> child)
BONES = [
    ("Pelvis",       "L5"),
    ("L5",           "L3"),
    ("L3",           "T12"),
    ("T12",          "T8"),
    ("T8",           "Neck"),
    ("Neck",         "Head"),
    ("T8",           "RightShoulder"),
    ("RightShoulder","RightUpperArm"),
    ("RightUpperArm","RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("T8",           "LeftShoulder"),
    ("LeftShoulder", "LeftUpperArm"),
    ("LeftUpperArm", "LeftForeArm"),
    ("LeftForeArm",  "LeftHand"),
    ("Pelvis",       "RightUpperLeg"),
    ("RightUpperLeg","RightLowerLeg"),
    ("RightLowerLeg","RightFoot"),
    ("RightFoot",    "RightToe"),
    ("Pelvis",       "LeftUpperLeg"),
    ("LeftUpperLeg", "LeftLowerLeg"),
    ("LeftLowerLeg", "LeftFoot"),
    ("LeftFoot",     "LeftToe"),
]


def _strip_ns(tag):
    return tag.split("}")[-1]


def parse_mvnx(path):
    tree = ET.parse(path)
    root = tree.getroot()

    seg_labels = None
    frames = []
    src_fps = None

    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag == "subject" and src_fps is None:
            fr = elem.get("frameRate")
            if fr:
                src_fps = float(fr)
        if tag == "segments" and seg_labels is None:
            seg_labels = [
                child.get("label")
                for child in elem
                if _strip_ns(child.tag) == "segment"
            ]
        if tag == "frame":
            pos = None
            for c in elem:
                if _strip_ns(c.tag) == "position" and c.text:
                    p = np.array(c.text.split(), dtype=np.float64)
                    pos = p.reshape(-1, 3)
            frames.append({
                "type": elem.get("type"),
                "pos": pos,
            })

    normal = [f for f in frames if f["type"] == "normal" and f["pos"] is not None]
    return seg_labels, normal, src_fps or 5.45455


def upsample(pos_arr, src_fps, tgt_fps):
    """pos_arr: (T, S, 3). Linear interpolate to tgt_fps."""
    T, S, _ = pos_arr.shape
    if tgt_fps <= src_fps or T < 2:
        return pos_arr, src_fps
    duration = (T - 1) / src_fps
    src_t = np.arange(T) / src_fps
    dst_t = np.linspace(0, duration, int(round(duration * tgt_fps)) + 1)
    out = np.zeros((len(dst_t), S, 3))
    for s in range(S):
        for k in range(3):
            out[:, s, k] = np.interp(dst_t, src_t, pos_arr[:, s, k])
    return out, tgt_fps


def make_video(mvnx_path, out_path, target_fps=30, elev=20, azim=45):
    seg_labels, frames, src_fps = parse_mvnx(mvnx_path)
    si = {l: i for i, l in enumerate(seg_labels)}

    pos_arr = np.stack([f["pos"] for f in frames])  # (T, 23, 3)
    pos_arr, actual_fps = upsample(pos_arr, src_fps, target_fps)
    T = len(pos_arr)

    # World bounds for fixed axes
    lo = pos_arr.reshape(-1, 3).min(0)
    hi = pos_arr.reshape(-1, 3).max(0)
    margin = 0.3
    xlim = (lo[0] - margin, hi[0] + margin)
    ylim = (lo[1] - margin, hi[1] + margin)
    zlim = (lo[2] - margin, hi[2] + margin)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    bone_lines = []
    for _ in BONES:
        line, = ax.plot([], [], [], "o-", color="steelblue", lw=2, ms=4)
        bone_lines.append(line)
    pelvis_dot, = ax.plot([], [], [], "ro", ms=8)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (up)")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.view_init(elev=elev, azim=azim)
    title = ax.set_title("")

    def update(t):
        pos = pos_arr[t]
        for line, (a, b) in zip(bone_lines, BONES):
            if a in si and b in si:
                pa, pb = pos[si[a]], pos[si[b]]
                line.set_data([pa[0], pb[0]], [pa[1], pb[1]])
                line.set_3d_properties([pa[2], pb[2]])
        pel = pos[si["Pelvis"]]
        pelvis_dot.set_data([pel[0]], [pel[1]])
        pelvis_dot.set_3d_properties([pel[2]])
        title.set_text(f"Frame {t}/{T-1}  |  {Path(mvnx_path).stem}")
        return bone_lines + [pelvis_dot, title]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000/actual_fps, blit=False)
    writer = animation.FFMpegWriter(fps=actual_fps, bitrate=2000)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved skeleton video -> {out_path}  ({T} frames @ {actual_fps:.1f} fps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mvnx", help="input .mvnx file")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--target-fps", type=float, default=30)
    ap.add_argument("--elev", type=float, default=20)
    ap.add_argument("--azim", type=float, default=45)
    args = ap.parse_args()
    out = args.out or str(Path(args.mvnx).with_suffix("_skeleton.mp4"))
    make_video(args.mvnx, out, target_fps=args.target_fps, elev=args.elev, azim=args.azim)
