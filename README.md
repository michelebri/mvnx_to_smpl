# MVNX → Booster T1 — Kick Retargeting

Convert Xsens **MVNX** motion-capture files into a video of the **Booster T1**
humanoid robot performing the same motion, via SMPL → SMPL-X → retargeting + MuJoCo rendering.

## Pipeline

```
MVNX   ──(1)──►  raw skeleton video       (visualize_mvnx.py)
MVNX   ──(2)──►  SMPL local_poses .npz    (mvnx_to_smpl.py)
SMPL   ──(3)──►  SMPL-X .npz              (smpl_to_smplx_npz.py)
SMPL-X ──(4)──►  Booster T1 video + pkl   (pipeline.py)
pkl    ──(5)──►  mjlab tracking npz        (gmr_to_tracking.py)
```

MVNX is exported at ~5.45 Hz and slerp-upsampled to 30 fps.

## Usage

```bash
pixi install   # first time only

pixi run retarget input/kick.mvnx   # single file
pixi run retarget input/             # whole folder
pixi run retarget input/ --skip-vis  # skip skeleton videos (faster)
```

Output per ogni file `<name>.mvnx` → `output/<name>/`:
```
output/<name>/skeleton.mp4      # step 1 – raw MVNX skeleton
output/<name>/smpl.npz          # step 2 – SMPL local_poses
output/<name>/smplx.npz         # step 3 – SMPL-X
output/<name>/t1.mp4            # step 4 – Booster T1 render
output/<name>/t1_gmr.pkl        # step 4 – retargeting output
output/<name>/t1_motion.npz     # step 5 – mjlab tracking format
```

## Layout

```
.
├── pipeline.py                # orchestrates all steps
├── mvnx_to_smpl.py            # (2) MVNX -> SMPL local_poses
├── smpl_to_smplx_npz.py       # (3) SMPL -> SMPL-X npz
├── gmr_to_tracking.py         # (5) pkl -> tracking npz
├── visualize_mvnx.py          # (1) raw MVNX skeleton video
├── replay_motion.py           # kinematic replay of a tracking npz
├── check_colosseum_npz.py     # validate a tracking npz
├── input/                     # source .mvnx files
├── output/                    # generated npz + mp4 (git-ignored)
├── models/smpl/SMPL_NEUTRAL.pkl
├── models/t1_23dof/           # T1_23dof MuJoCo model
├── GMR/
└── articulate/
```

## Models (not in git)

- `models/smpl/SMPL_NEUTRAL.pkl` — from [SMPL](https://smpl.is.tue.mpg.de/)
- `GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz` — from [SMPL-X](https://smpl-x.is.tue.mpg.de/)

## Setup

```bash
pixi install
```
