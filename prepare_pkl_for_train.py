"""
This module provides functionality to convert motion data from GMR format to MimicKit format. 

Usage:
    Command line:
        python tools/data_format/gmr_to_mimickit.py
    Required arguments:
        --input_file PATH       Path to the input GMR pickle file
        --output_file PATH      Path to save the output MimicKit pickle file
    Optional arguments:
        --loop {wrap,clamp}     Loop mode for the motion (default: wrap)
        --start_frame INT       Start frame for motion clipping (default: 0)
        --end_frame INT         End frame for motion clipping (default: -1, uses all frames)
        --output_fps INT        Frame rate for the output motion (default: same as input)
    
GMR Format:
    The input GMR format should be a pickle file containing a dictionary with keys:
    - 'fps': Frame rate (int)
    - 'root_pos': Root position array, shape (num_frames, 3)
    - 'root_rot': Root rotation quaternions, shape (num_frames, 4), format (x, y, z, w)
    - 'dof_pos': Degrees of freedom positions, shape (num_frames, num_dofs)
    - 'local_body_pos': Currently unused (can be None)
    - 'link_body_list': Currently unused (can be None)

Output:
    Creates a dictionary containing MimicKit motion data saved as a pickle file, with loop mode stored as INT and motion data stored as
    concatenated arrays of [root_pos, root_rot_expmap, dof_pos] per frame.
"""

import argparse
import pickle
import numpy as np
import sys

import torch

sys.path.append(".")  # Ensure the repository root is on sys.path so we can use some utilities.

from utils_from_mimickit import Motion, LoopMode, quat_to_exp_map

def convert_gmr_to_mimickit(gmr_file_path, output_file_path, loop_mode, start_frame, end_frame, output_fps):
    """
    Convert a GMR compatible motion dataset to MimicKit compatible dataset.
    
    Args:
        gmr_file_path (str): Path to the GMR format pickle file
        output_file_path (str): Path to save the MimicKit format pickle file
        loop_mode (bool): Whether the motion should loop (Set to wrap as default)
    """
    if loop_mode == "wrap":
        loop_mode_out = LoopMode.WRAP # MimicKit LoopMode.WRAP
    elif loop_mode == "clamp":
        loop_mode_out = LoopMode.CLAMP # MimicKit LoopMode.CLAMP
    else:
        raise ValueError(f"Invalid loop_mode: {loop_mode}. Choose 'wrap' or 'clamp'.")
    
    # Load GMR format data
    with open(gmr_file_path, 'rb') as f:
        gmr_data = pickle.load(f)
    
    # Extract data from GMR format
    fps = gmr_data['fps']
    root_pos = gmr_data['root_pos']  # Shape: (num_frames, 3)
    root_rot_quat = gmr_data['root_rot']  # Shape: (num_frames, 4), quaternion format
    dof_pos = gmr_data['dof_pos']    # Shape: (num_frames, num_dofs)

    # Log the type and shape of each extracted term
    print("\n" + "="*60)
    print("📥 LOADED GMR DATA")
    print("="*60)
    print(f"⏱️  FPS:           type={type(fps).__name__}, value={fps}")
    print(f"📍 Root Position: type={type(root_pos).__name__}, shape={root_pos.shape}")
    print(f"🔄 Root Rotation: type={type(root_rot_quat).__name__}, shape={root_rot_quat.shape}")
    print(f"🦴 DOF Position:  type={type(dof_pos).__name__}, shape={dof_pos.shape}")
    print("="*60 + "\n")
    
    # Verify shapes
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos shape (num_frames, 3), got {root_pos.shape}")
        
    if root_rot_quat.ndim != 2 or root_rot_quat.shape[1] != 4:
        raise ValueError(f"Expected root_rot_quat shape (num_frames, 4), got {root_rot_quat.shape}")
        
    if dof_pos.ndim != 2:
        raise ValueError(f"Expected dof_pos to be 2D array, got {dof_pos.ndim}D")

    # Convert quaternion to exponential map
    root_rot = quat_to_exp_map(torch.tensor(root_rot_quat)).numpy()  
    
    # Stack all motion data along the last dimension
    # frames shape: (num_frames, 3 + 3 + num_dofs) = (num_frames, 6 + num_dofs)
    frames = np.concatenate([root_pos, root_rot, dof_pos], axis=1)

    # Chop frames
    if end_frame == -1:
        end_frame = frames.shape[0]
    assert 0 <= start_frame < end_frame <= frames.shape[0], "Invalid start_frame or end_frame."
    frames = frames[start_frame:end_frame, :]

    save_fps = fps if output_fps == -1 else output_fps

    out_data = Motion(loop_mode=loop_mode_out, fps=save_fps, frames=frames)

    # Save to MimicKit format
    out_data.save(output_file_path)
    
    print("\n" + "="*60)
    print("✅ CONVERSION SUCCESSFUL")
    print("="*60)
    print(f"📁 Input:  {gmr_file_path}")
    print(f"💾 Output: {output_file_path}")
    print("-"*60)
    print(f"📊 Frames Shape:  {frames.shape}")
    print(f"🎬 Total Frames: {frames.shape[0]}")
    print(f"⏱️  FPS:          {save_fps}")
    print(f"🔄 Loop Mode:    {loop_mode_out}")
    print("="*60 + "\n")

    return out_data



"""Convert MimicKit pickle motion files to CSV format."""

import pickle

import numpy as np
import tyro
from scipy.spatial.transform import Rotation, Slerp


class FlexibleClass:
  """A class that accepts any arguments and stores them."""

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
  """Cubic ease-out curve for smooth deceleration."""
  return 1 - (1 - t) ** 3


def ease_in_cubic(t):
  """Cubic ease-in curve for smooth acceleration."""
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
  """Convert pickle motion file to CSV format.

  Output format: [root_pos(3), quat_xyzw(4), joint_dof(N)]

  Args:
    pkl_file: Path to input .pkl file.
    csv_file: Path to output .csv file.
    duration: Desired duration in seconds. If None, use original duration. Motion will
      be cycled to reach this duration.
    pad_duration: Duration in seconds to hold the final pose at the end.
    transition_duration: Duration in seconds to blend to/from safe standing pose.
    add_start_transition: Whether to add transition from safe standing pose to motion start.
    add_end_transition: Whether to add transition from motion end to safe standing pose.
  """
  # Hardcoded safe standing pose joints.
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

  SAFE_Z_HEIGHT = 0.76  # Safe standing height.

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

  print(f"Loaded motion with shape: {frames.shape}")
  print(f"FPS: {fps}")
  print(
    f"Original duration: {original_duration:.2f} seconds ({frames.shape[0]} frames)"
  )

  # Repeat motion if duration is specified.
  if duration is not None:
    if duration < original_duration:
      print(
        f"Warning: Requested duration ({duration}s) is shorter than original ({original_duration:.2f}s)"
      )
      print("         Truncating motion...")
      num_frames = int(duration * fps)
      frames = frames[:num_frames]
    else:
      num_cycles = int(np.ceil(duration / original_duration))
      print(f"Repeating motion {num_cycles} times to reach {duration}s...")

      # Calculate displacement per cycle (only XY, not Z).
      start_pos = frames[0, 0:3]
      end_pos = frames[-1, 0:3]
      displacement_per_cycle = end_pos - start_pos

      # Only accumulate XY displacement, zero out Z.
      displacement_per_cycle[2] = 0.0

      print(
        f"XY displacement per cycle: [{displacement_per_cycle[0]:.4f}, {displacement_per_cycle[1]:.4f}]"
      )
      if np.linalg.norm(displacement_per_cycle[:2]) < 1e-3:
        print("  (In-place motion detected)")
      else:
        print("  (Forward motion detected - accumulating XY displacement only)")

      # Repeat and accumulate XY displacement only.
      repeated_frames = []
      for cycle in range(num_cycles):
        cycle_frames = frames.copy()
        # Add cumulative displacement to this cycle (XY only).
        cycle_frames[:, 0:3] += displacement_per_cycle * cycle
        repeated_frames.append(cycle_frames)

      frames = np.vstack(repeated_frames)

      # Truncate to exact duration.
      num_frames = int(duration * fps)
      frames = frames[:num_frames]

    print(f"New motion shape: {frames.shape}")
    print(
      f"New duration: {frames.shape[0] / fps:.2f} seconds ({frames.shape[0]} frames)"
    )

  # Add transition FROM safe standing pose TO motion start.
  if add_start_transition:
    print("\nAdding start transition from safe standing pose to motion start...")

    # Get the first frame as target.
    target_frame = frames[0].copy()

    # Number of transition frames.
    transition_frames = int(transition_duration * fps)
    print(
      f"Creating {transition_duration}s start transition ({transition_frames} frames)..."
    )

    # Extract components from target (first) frame.
    target_pos = target_frame[0:3]
    target_rot = target_frame[3:6]  # axis-angle
    target_joints = target_frame[6:]

    # Convert target rotation to euler angles to extract yaw.
    target_rot_obj = Rotation.from_rotvec(target_rot)
    euler_zyx = target_rot_obj.as_euler("ZYX", degrees=False)
    yaw = euler_zyx[0]

    print(
      f"  First frame orientation: yaw={np.degrees(yaw):.1f}°, pitch={np.degrees(euler_zyx[1]):.1f}°, roll={np.degrees(euler_zyx[2]):.1f}°"
    )
    print(
      f"  Start orientation: yaw={np.degrees(yaw):.1f}° (matched), pitch=0°, roll=0°"
    )

    # Create start rotation: yaw from first frame, roll=0, pitch=0.
    start_rot_obj = Rotation.from_euler("ZYX", [yaw, 0, 0], degrees=False)
    start_rot = start_rot_obj.as_rotvec()

    # Start position: XY from first frame, Z at safe height.
    start_pos = target_pos.copy()
    start_pos[2] = SAFE_Z_HEIGHT

    # Start joints: safe standing pose.
    start_joints = SAFE_POSE_JOINTS

    # Create Slerp interpolator for smooth rotation transition.
    key_times = [0, 1]
    key_rots = Rotation.concatenate([start_rot_obj, target_rot_obj])
    slerp = Slerp(key_times, key_rots)

    # Create transition frames.
    start_transition = []
    for i in range(transition_frames):
      t = i / (transition_frames - 1) if transition_frames > 1 else 1.0
      t_eased = ease_in_cubic(t)  # Apply ease-in for acceleration.

      # LERP position.
      pos = start_pos * (1 - t_eased) + target_pos * t_eased

      # SLERP rotation.
      rot_obj = slerp(t_eased)
      rot = rot_obj.as_rotvec()

      # LERP joints.
      joints = start_joints * (1 - t_eased) + target_joints * t_eased

      # Combine.
      frame = np.concatenate([pos, rot, joints])
      start_transition.append(frame)

    start_transition = np.array(start_transition)
    frames = np.vstack([start_transition, frames])
    print(f"After start transition, motion shape: {frames.shape}")
    print(f"Total duration so far: {frames.shape[0] / fps:.2f} seconds")

  # Add transition TO safe standing pose at the end.
  if add_end_transition:
    print(
      "\nAdding end transition to safe standing pose (keeping yaw, resetting roll/pitch)..."
    )

    # Get the last frame as starting point for transition.
    start_frame = frames[-1].copy()

    # Number of transition frames.
    transition_frames = int(transition_duration * fps)
    print(
      f"Creating {transition_duration}s end transition ({transition_frames} frames)..."
    )

    # Extract components from start frame.
    start_pos = start_frame[0:3]
    start_rot = start_frame[3:6]  # axis-angle
    start_joints = start_frame[6:]

    # Convert start rotation to euler angles to extract yaw.
    start_rot_obj = Rotation.from_rotvec(start_rot)
    euler_zyx = start_rot_obj.as_euler("ZYX", degrees=False)
    yaw = euler_zyx[0]  # Extract yaw.

    print(
      f"  Final frame orientation: yaw={np.degrees(yaw):.1f}°, pitch={np.degrees(euler_zyx[1]):.1f}°, roll={np.degrees(euler_zyx[2]):.1f}°"
    )
    print(f"  Target orientation: yaw={np.degrees(yaw):.1f}° (kept), pitch=0°, roll=0°")

    # Create target rotation: yaw from final frame, roll=0, pitch=0.
    target_rot_obj = Rotation.from_euler("ZYX", [yaw, 0, 0], degrees=False)
    target_rot = target_rot_obj.as_rotvec()

    # Target: keep XY position, transition Z to safe height.
    target_pos = start_pos.copy()
    target_pos[2] = SAFE_Z_HEIGHT

    # Target joints from safe pose.
    target_joints = SAFE_POSE_JOINTS

    # Create Slerp interpolator for smooth rotation transition.
    key_times = [0, 1]
    key_rots = Rotation.concatenate([start_rot_obj, target_rot_obj])
    slerp = Slerp(key_times, key_rots)

    # Create transition frames.
    end_transition = []
    for i in range(transition_frames):
      t = i / (transition_frames - 1) if transition_frames > 1 else 1.0
      t_eased = ease_out_cubic(t)  # Apply ease-out for deceleration.

      # LERP position.
      pos = start_pos * (1 - t_eased) + target_pos * t_eased

      # SLERP rotation.
      rot_obj = slerp(t_eased)
      rot = rot_obj.as_rotvec()

      # LERP joints.
      joints = start_joints * (1 - t_eased) + target_joints * t_eased

      # Combine.
      frame = np.concatenate([pos, rot, joints])
      end_transition.append(frame)

    end_transition = np.array(end_transition)
    frames = np.vstack([frames, end_transition])
    print(f"After end transition, motion shape: {frames.shape}")
    print(f"Total duration: {frames.shape[0] / fps:.2f} seconds")

  # Add padding at the end (hold final pose).
  if pad_duration > 0:
    pad_frames = int(pad_duration * fps)
    print(
      f"\nAdding {pad_duration}s padding ({pad_frames} frames) - holding final pose..."
    )

    # Repeat the last frame.
    final_frame = frames[-1:].copy()
    padding = np.tile(final_frame, (pad_frames, 1))

    frames = np.vstack([frames, padding])
    print(f"Padded motion shape: {frames.shape}")
    print(
      f"Total duration: {frames.shape[0] / fps:.2f} seconds ({frames.shape[0]} frames)"
    )

  # Parse the data.
  root_pos = frames[:, 0:3]  # (N, 3)
  root_rot_3d = frames[:, 3:6]  # (N, 3) - axis-angle
  joint_dof = frames[:, 6:]  # (N, 29)

  print(f"\nRoot position: {root_pos.shape}")
  print(f"Root rotation (axis-angle): {root_rot_3d.shape}")
  print(f"Joint DOF: {joint_dof.shape}")

  # Convert axis-angle rotation to quaternion (XYZW format).
  print("\nConverting rotations from axis-angle to quaternions (XYZW)...")
  quats = []
  for rot_vec in root_rot_3d:
    angle = np.linalg.norm(rot_vec)
    if angle > 1e-6:
      rotation = Rotation.from_rotvec(rot_vec)
    else:
      rotation = Rotation.from_quat([0, 0, 0, 1])

    # Get quaternion in XYZW format.
    quat_xyzw = rotation.as_quat()  # Returns [x, y, z, w]
    quats.append(quat_xyzw)

  quats = np.array(quats)  # (N, 4) in XYZW format.
  print(f"Quaternions (XYZW): {quats.shape}")

  # Combine: [root_pos(3), quat_xyzw(4), joint_dof(29)].
  csv_data = np.concatenate([root_pos, quats, joint_dof], axis=1)
  print(
    f"Final CSV shape: {csv_data.shape} (columns: 3 pos + 4 quat_xyzw + {joint_dof.shape[1]} joints)"
  )

  # Save to CSV.
  print(f"\nSaving to {csv_file}...")
  np.savetxt(csv_file, csv_data, delimiter=",", fmt="%.8f")
  print("Done!")

  # Print some stats.
  print("\nMotion stats:")
  print(f"  Duration: {csv_data.shape[0] / fps:.2f} seconds")
  print(f"  Frames: {csv_data.shape[0]}")
  print(f"  FPS: {fps}")
  print("\nOutput format: [x, y, z, qx, qy, qz, qw, joint1, joint2, ...]")



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