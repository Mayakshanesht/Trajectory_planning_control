# Realistic MPC Tracking (Student Version)

This project demonstrates trajectory tracking with a linearized Model Predictive Controller (MPC) for a kinematic bicycle model.

## What is new
- Upload-based reference input (`.csv`/`.xlsx`) with validation (`x,y` or `s,offset` format).
- Guided anchor-offset mode: set lateral offsets at `START`, `+peak`, `-peak`, `END`.
- Curvature-aware reference speed scaling for improved zigzag/lane-change tracking.
- Refactored `control/mpc.py` so students can clearly see:
  - objective terms,
  - system model constraints,
  - input constraints,
  - road corridor constraints,
  - QP assembly and OSQP solve.

## Quick start (Conda recommended)
```bash
conda create -n mpc-tracking python=3.12 -y
conda activate mpc-tracking
python -m pip install --upgrade pip
pip install -r requirements.txt
python launcher.py --interface web
```

Then open the Gradio URL printed in terminal (default is usually `http://127.0.0.1:7860`).

## Reinstall in the Conda env (if dependencies got mixed)
```bash
conda activate mpc-tracking
pip install --upgrade --force-reinstall -r requirements.txt
python launcher.py --interface web
```

## Frontend workflow (no mouse required)
Set common MPC controls first:
1. Choose `Reference speed` and `MPC horizon`.

Then pick one workflow:
1. `Predefined References`: quick benchmark for ref1/ref2/ref3.
2. `Upload Reference`: upload file with trajectory points.
3. `Guided Anchor Offsets`: provide 4 lateral offsets at fixed anchor locations.

### Upload reference format
1. Open `Upload Reference` tab.
2. Click `Generate CSV Template`.
3. Fill either:
   - `x,y` columns (meters), or
   - `s,offset` where `s in [0,1]` and `offset` is lateral offset from centerline.
4. Upload the file and click `Run MPC (Uploaded Reference)`.

### Guided anchor mode
1. Open `Guided Anchor Offsets`.
2. Set lateral offsets at fixed anchors: `START`, first positive peak, first negative peak, `END`.
3. Run `Run MPC (Anchor Offsets)`.

## MPC formulation used
At each control step, we solve a QP over horizon `N`:

- Objective:
  - state tracking: `sum (x_k - x_ref,k)^T Q (x_k - x_ref,k)`
  - input effort: `sum u_k^T R u_k`
  - input smoothness: `sum (u_{k+1} - u_k)^T Rj (u_{k+1} - u_k)`

- Constraints:
  - linearized model: `x_{k+1} = A_k x_k + B_k u_k + c_k`
  - actuator bounds: `u_min <= u_k <= u_max`
  - track corridor half-spaces from road geometry.

Implementation entry points:
- QP construction and solve: `control/mpc.py`
- Linearization: `control/linearize.py`
- Dynamics model: `vehicle/dynamics.py`
- Corridor constraints: `track/track.py`

## Student map of optimizer matrices
Inside `control/mpc.py`:
- `_build_cost(...)` creates `P, q`.
- `_build_dynamics_constraints(...)` adds model equalities.
- `_build_input_constraints(...)` adds control bounds.
- `_build_jerk_constraints(...)` links `delta_u = u_{k+1} - u_k`.
- `_build_track_constraints(...)` adds road limits.
- `solve(...)` stacks all blocks into `A, l, u` and calls OSQP.

## CLI mode
```bash
python launcher.py --interface cli
```

## Key files
- `web_interface.py`: structured web UI + simulation entry points.
- `control/mpc.py`: educational MPC QP assembly.
- `main_simple.py`: scriptable MPC runs for predefined references.
- `utils/plotting.py`: GIF generation.
- `docs/MPC_IMPLEMENTATION_TEACHING_GUIDE.md`: step-by-step implementation guide for teaching/slides.
