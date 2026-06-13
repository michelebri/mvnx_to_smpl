"""
Convert retargeting pkl to training pkl + CSV.

Usage:
    python prepare_pkl_for_train.py --input_file <in.pkl> --output_file <out.pkl>
    Optional: --loop {wrap,clamp}, --start_frame INT, --end_frame INT, --output_fps INT
"""

import argparse
import pickle
import numpy as np
import sys

import torch

sys.path.append(".")

from utils_from_mimickit import Motion, LoopMode, quat_to_exp_map

def convert_gmr_to_mimickit(gmr_file_path, output_file_path, loop_mode, start_frame, end_frame, output_fps):
    if loop_mode == "wrap":
        loop_mode_out = LoopMode.WRAP
    elif loop_mode == "clamp":
        loop_mode_out = LoopMode.CLAMP
    else:
        raise ValueError(f"Invalid loop_mode: {loop_mode}. Choose 'wrap' or 'clamp'.")

    with open(gmr_file_path, 'rb') as f:
        gmr_data = pickle.load(f)

    fps = gmr_data['fps']
    root_pos = gmr_data['root_pos']
    root_rot_quat = gmr_data['root_rot']
    dof_pos = gmr_data['dof_pos']

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos shape (num_frames, 3), got {root_pos.shape}")
    if root_rot_quat.ndim != 2 or root_rot_quat.shape[1] != 4:
        raise ValueError(f"Expected root_rot_quat shape (num_frames, 4), got {root_rot_quat.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"Expected dof_pos to be 2D array, got {dof_pos.ndim}D")

    root_rot = quat_to_exp_map(torch.tensor(root_rot_quat)).numpy()
    frames = np.concatenate([root_pos, root_rot, dof_pos], axis=1)

    if end_frame == -1:
        end_frame = frames.shape[0]
    assert 0 <= start_frame < end_frame <= frames.shape[0], "Invalid start_frame or end_frame."
    frames = frames[start_frame:end_frame, :]

    save_fps = fps if output_fps == -1 else output_fps
    out_data = Motion(loop_mode=loop_mode_out, fps=save_fps, frames=frames)
    out_data.save(output_file_path)

    print(f"Saved {output_file_path}  ({frames.shape[0]} frames @ {save_fps} fps)")
    return out_data


import pickle

import numpy as np
import tyro
from scipy.spatial.transform import Rotation, Slerp


class FlexibleClass:
  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs
    for k, v in kwargs.items():
      setattr(self, k, v)

  def __repr__(self):
    attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
    return f"{self.__class__.__name__}({list(attrs.keys())})"


class RobustUnpickler(pickle.Unpickler):
  def find_class(self, module, name):
    try:
      return super().find_class(module, name)
    except (ImportError, AttributeError):
      return type(name, (FlexibleClass,), {})


def ease_out_cubic(t):
  return 1 - (1 - t) ** 3


def ease_in_cubic(t):
  return t**3


def main(
  pkl_file: str,
  csv_file: str,
  duration: float | None = None,
  pad_duration: float = 0.0,
  transition_duration: float = 1.0,
  add_start_transition: bool = False,
  add_end_transition: bool = False,
):
  """Convert pickle motion file to CSV. Output: [root_pos(3), quat_xyzw(4), joint_dof(N)]"""
  # fmt: off
  SAFE_POSE_JOINTS = np.array(
    [
      -0.312, 0, 0, 0.669, -0.363, 0,  # left leg
      -0.312, 0, 0, 0.669, -0.363, 0,  # right leg
      0, 0, 0,  # waist
      0.2, 0.2, 0, 0.6, 0, 0, 0,  # left arm
      0.2, -0.2, 0, 0.6, 0, 0, 0,  # right arm
    ]
  )
  # fmt: on

  SAFE_Z_HEIGHT = 0.76

  print(f"Loading {pkl_file}...")
  with open(pkl_file, "rb") as f:
    unpickler = RobustUnpickler(f)
    data = unpickler.load()

  if isinstance(data, dict):
    frames = data["frames"]
    fps = data["fps"]
  else:
    frames = data.frames
    fps = data.fps

  frames = np.asarray(frames)
  original_duration = frames.shape[0] / fps

  print(f"Loaded motion: {frames.shape}, {fps} fps, {original_duration:.2f}s")

  if duration is not None:
    if duration < original_duration:
      print(f"Warning: Truncating {original_duration:.2f}s -> {duration}s")
      frames = frames[:int(duration * fps)]
    else:
      num_cycles = int(np.ceil(duration / original_duration))
      print(f"Repeating motion {num_cycles} times to reach {duration}s...")

      displacement_per_cycle = frames[-1, 0:3] - frames[0, 0:3]
      displacement_per_cycle[2] = 0.0

      repeated_frames = []
      for cycle in range(num_cycles):
        cycle_frames = frames.copy()
        cycle_frames[:, 0:3] += displacement_per_cycle * cycle
        repeated_frames.append(cycle_frames)

      frames = np.vstack(repeated_frames)[:int(duration * fps)]

    print(f"New shape: {frames.shape}, {frames.shape[0] / fps:.2f}s")

  if add_start_transition:
    target_frame = frames[0].copy()
    transition_frames = int(transition_duration * fps)

    target_pos = target_frame[0:3]
    target_rot = target_frame[3:6]
    target_joints = target_frame[6:]

    target_rot_obj = Rotation.from_rotvec(target_rot)
    euler_zyx = target_rot_obj.as_euler("ZYX", degrees=False)
    yaw = euler_zyx[0]

    start_rot_obj = Rotation.from_euler("ZYX", [yaw, 0, 0], degrees=False)
    start_rot = start_rot_obj.as_rotvec()
    start_pos = target_pos.copy()
    start_pos[2] = SAFE_Z_HEIGHT
    start_joints = SAFE_POSE_JOINTS

    slerp = Slerp([0, 1], Rotation.concatenate([start_rot_obj, target_rot_obj]))

    start_transition = []
    for i in range(transition_frames):
      t = i / (transition_frames - 1) if transition_frames > 1 else 1.0
      t_eased = ease_in_cubic(t)
      pos = start_pos * (1 - t_eased) + target_pos * t_eased
      rot = slerp(t_eased).as_rotvec()
      joints = start_joints * (1 - t_eased) + target_joints * t_eased
      start_transition.append(np.concatenate([pos, rot, joints]))

    frames = np.vstack([np.array(start_transition), frames])
    print(f"After start transition: {frames.shape}")

  if add_end_transition:
    start_frame = frames[-1].copy()
    transition_frames = int(transition_duration * fps)

    start_pos = start_frame[0:3]
    start_rot = start_frame[3:6]
    start_joints = start_frame[6:]

    start_rot_obj = Rotation.from_rotvec(start_rot)
    euler_zyx = start_rot_obj.as_euler("ZYX", degrees=False)
    yaw = euler_zyx[0]

    target_rot_obj = Rotation.from_euler("ZYX", [yaw, 0, 0], degrees=False)
    target_pos = start_pos.copy()
    target_pos[2] = SAFE_Z_HEIGHT
    target_joints = SAFE_POSE_JOINTS

    slerp = Slerp([0, 1], Rotation.concatenate([start_rot_obj, target_rot_obj]))

    end_transition = []
    for i in range(transition_frames):
      t = i / (transition_frames - 1) if transition_frames > 1 else 1.0
      t_eased = ease_out_cubic(t)
      pos = start_pos * (1 - t_eased) + target_pos * t_eased
      rot = slerp(t_eased).as_rotvec()
      joints = start_joints * (1 - t_eased) + target_joints * t_eased
      end_transition.append(np.concatenate([pos, rot, joints]))

    frames = np.vstack([frames, np.array(end_transition)])
    print(f"After end transition: {frames.shape}")

  if pad_duration > 0:
    pad_frames = int(pad_duration * fps)
    frames = np.vstack([frames, np.tile(frames[-1:], (pad_frames, 1))])
    print(f"After padding: {frames.shape}")

  root_pos = frames[:, 0:3]
  root_rot_3d = frames[:, 3:6]
  joint_dof = frames[:, 6:]

  quats = []
  for rot_vec in root_rot_3d:
    angle = np.linalg.norm(rot_vec)
    rotation = Rotation.from_rotvec(rot_vec) if angle > 1e-6 else Rotation.from_quat([0, 0, 0, 1])
    quats.append(rotation.as_quat())

  quats = np.array(quats)
  csv_data = np.concatenate([root_pos, quats, joint_dof], axis=1)

  np.savetxt(csv_file, csv_data, delimiter=",", fmt="%.8f")
  print(f"Saved {csv_file}  shape={csv_data.shape}, {csv_data.shape[0] / fps:.2f}s @ {fps} fps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR motion data to MimicKit format.")
    parser.add_argument("--input_file", required=True, help="Path to the input GMR pickle file")
    parser.add_argument("--output_file", required=True, help="Path to the output MimicKit pickle file")
    parser.add_argument("--loop", default="wrap", choices=["wrap", "clamp"], help="Enable loop mode on the converted motion")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for chopping the motion")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame for chopping the motion")
    parser.add_argument("--output_fps", type=int, default=-1, help="Frame rate for the output motion (default: same as input)")
    args = parser.parse_args()

    convert_gmr_to_mimickit(args.input_file, args.output_file, loop_mode=args.loop, start_frame=args.start_frame, end_frame=args.end_frame, output_fps=args.output_fps)
    main(pkl_file=args.output_file, csv_file=args.output_file.replace('.pkl', '.csv'))
