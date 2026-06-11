# MVNX → Booster T1 — Kick Retargeting

Convert Xsens **MVNX** motion-capture files into a video of the **Booster T1**
humanoid robot performing the same motion (e.g. a kick), via SMPL → SMPL-X →
[GMR](https://github.com/YanjieZe/GMR) retargeting + MuJoCo rendering.

## Pipeline

```
MVNX  ──(1)──►  raw skeleton video        (visualize_mvnx.py)
MVNX  ──(2)──►  SMPL  local_poses .npz     (mvnx_to_smpl.py)
SMPL  ──(3)──►  SMPL skeleton video        (visualize_smpl.py)   [sanity check]
SMPL  ──(4)──►  SMPL-X .npz                (smpl_to_smplx_npz.py)
SMPL-X ─(5)──►  Booster T1 video           (render_t1.py, via GMR/)
```

Steps 1 and 3 are visual sanity checks; the production path is 2 → 4 → 5.

MVNX is exported at ~5.45 Hz and slerp-upsampled to 30 fps inside `mvnx_to_smpl.py`.

## Usage

```bash
pixi install   # first time only — sets up the environment

# single file
pixi run retarget "input/kick_flavio-004#MVN System 2.mvnx"

# whole folder
pixi run retarget input/

# skip sanity-check videos (faster)
pixi run retarget input/ --skip-vis
```

Output per ogni file `<name>.mvnx` → `output/<name>/`:
```
output/<name>/skeleton.mp4   # step 1 – raw MVNX skeleton
output/<name>/smpl.npz       # step 2 – SMPL local_poses
output/<name>/smpl_vis.mp4   # step 3 – SMPL skeleton sanity check
output/<name>/smplx.npz      # step 4 – SMPL-X for GMR
output/<name>/t1.mp4         # step 5 – Booster T1 render
```

## Layout

```
.
├── mvnx_to_smpl.py            # (2) MVNX -> SMPL local_poses  (articulate IK)
├── smpl_to_smplx_npz.py       # (4) SMPL -> SMPL-X npz for GMR
├── render_t1.py               # (5) SMPL-X -> Booster T1 video (uses GMR/)
├── visualize_mvnx.py          # (1) raw MVNX skeleton video
├── visualize_smpl.py          # (3) SMPL skeleton video
├── input/                     # source .mvnx files
├── output/                    # generated npz + mp4 (git-ignored)
├── models/smpl/SMPL_NEUTRAL.pkl          # SMPL model (git-ignored, see below)
├── GMR/                       # vendored General Motion Retargeting (booster_t1 only)
│   └── assets/body_models/smplx/SMPLX_NEUTRAL.npz   # (git-ignored)
└── articulate/                # SMPL IK/FK library (third-party, see Credits)
```

## Key implementation notes

Three coordinate/grounding fixes are essential and easy to get wrong:

1. **Quaternion permutation** (`convert_coord` in `mvnx_to_smpl.py`): Xsens is
   Z-up, SMPL is Y-up. Use `new_x=old_y, new_y=old_z, new_z=old_x` (matching the
   reference `preprocess_mvnx_to_smpl.py` from
   [nawta/nymeria_smpl_processor](https://github.com/nawta/nymeria_smpl_processor)).
   A wrong permutation makes the torso drift forward and never recover (robot looks
   like it falls into a push-up).

2. **Y-up → Z-up** (`render_t1.py`): after `get_smplx_data_offline_fast`, rotate
   every joint position/orientation by `R = [[1,0,0],[0,0,-1],[0,1,0]]`. GMR's
   SMPL-X loader does *not* do this (only its GVHMR loader does), and MuJoCo is
   Z-up — without it the robot lies flat.

3. **Feet on ground** (`render_t1.py`): call `retarget(frame, offset_to_ground=True)`
   so GMR lifts the body each frame to keep the lowest foot on the floor. Without
   it the pelvis sinks during the kick and the robot bends its knees.

## Models (not in git)

Large model weights are git-ignored. Provide them at these paths:

- `models/smpl/SMPL_NEUTRAL.pkl` — SMPL neutral model (used by the articulate IK).
- `GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz` — SMPL-X neutral model
  (used by GMR for forward kinematics).

Get them from the official [SMPL](https://smpl.is.tue.mpg.de/) /
[SMPL-X](https://smpl-x.is.tue.mpg.de/) sites (registration required).

## Setup

```bash
pixi install   # creates the env (torch+cuda, mujoco, smplx, mink, …)
```

Then provide the model weights (git-ignored, see **Models** section above).

## Credits

- **`articulate/`** — SMPL IK/FK library, from
  [nawta/nymeria_smpl_processor](https://github.com/nawta/nymeria_smpl_processor/tree/main/mvnx_to_smpl/core/articulate).
- **`GMR/`** — [General Motion Retargeting](https://github.com/YanjieZe/GMR)
  (only the `booster_t1` assets are kept).
