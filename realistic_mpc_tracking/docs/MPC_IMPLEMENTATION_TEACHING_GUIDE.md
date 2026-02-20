# MPC Implementation Teaching Guide

Audience: students who understand MPC theory but want to implement it in code.

Use this as a step-by-step lesson plan or slide outline.

## 1. Learning Goals
- Map each MPC equation to concrete Python code.
- Understand how a trajectory reference is converted into optimizer inputs.
- Read and modify a QP-based MPC implementation safely.

## 2. System Overview (Slide: Architecture)
Pipeline in this repo:
1. Build road and reference trajectory.
2. Linearize vehicle dynamics along the reference.
3. Build QP cost + constraints.
4. Solve QP with OSQP.
5. Apply first control input.
6. Repeat in receding horizon loop.

Core files:
- `control/mpc.py`
- `control/linearize.py`
- `vehicle/dynamics.py`
- `track/track.py`
- `web_interface.py`

## 3. State, Input, and Horizon (Slide: Problem Setup)
State vector used in the project:
- `x = [X, Y, psi, vx, vy, r]`

Control vector:
- `u = [steer, accel]`

Horizon:
- `N` from UI slider (`MPC horizon`).

## 4. QP Formulation in Code (Slides: Cost and Constraints)
In `control/mpc.py`:
- `_build_cost(...)` builds matrix `P` and vector `q`.
- `_build_dynamics_constraints(...)` enforces `x_{k+1}=A_k x_k + B_k u_k + c_k`.
- `_build_input_constraints(...)` applies actuator bounds.
- `_build_jerk_constraints(...)` defines `u_{k+1}-u_k` smoothing variables.
- `_build_track_constraints(...)` enforces road corridor half-space limits.
- `solve(...)` stacks all blocks into `A, l, u` and calls OSQP.

Teaching point:
- Show students how each term in the objective appears as a block in `P`.

## 5. Linearization Step (Slide: LTV MPC)
In `control/linearize.py`:
- Dynamics are linearized at each horizon step around current reference and prior input.
- Output is `(A_k, B_k, c_k)` for each `k`.

Teaching point:
- Explain why this is Linear Time-Varying (LTV), not time-invariant.

## 6. Receding Horizon Loop (Slide: MPC Runtime Loop)
In `web_interface.py`, runtime loop:
1. Find current progress index on reference.
2. Slice next `N` reference states.
3. Build linearized model list `(A,B,c)`.
4. Build track corridor constraints.
5. Solve QP.
6. Apply only first control action.
7. Simulate one step and repeat.

Teaching point:
- MPC solves for a sequence, executes one action, re-solves next step.

## 7. Reference Generation (Slide: Inputs to MPC)
Available modes:
- Predefined references (`ref1`, `ref2`, `ref3`).
- Uploaded file (`x,y` or `s,offset`).
- Guided anchor offsets (4 anchors with cubic fit).

Implementation details:
- Reference path is converted to `[x, y, psi, v, 0, 0]`.
- Speed profile is curvature-aware (slower at high curvature).

Teaching point:
- Better reference quality usually improves controller behavior more than retuning alone.

## 8. Typical Failure Modes (Slide: Debug Checklist)
If tracking looks poor:
1. Reference too aggressive (sharp lateral change).
2. Speed too high for curvature.
3. Horizon too short.
4. Weight imbalance (`Q`, `R`, `Rj`).
5. Actuator constraints too tight.

Where to inspect:
- `control/mpc.py` for optimization setup.
- `web_interface.py` for reference shaping and runtime loop.

## 9. Parameter Tuning Exercise (Slide: Lab)
Suggested exercise:
1. Fix reference to `ref2` (zigzag).
2. Sweep horizon: `20, 30, 40, 50`.
3. Sweep speed: `2, 3, 4, 5 m/s`.
4. Record:
   - final error
   - average error
   - max error
   - completion progress

Discuss:
- Which parameter has bigger impact and why?

## 10. Minimal Implementation Path (Slide: Build From Scratch)
If students implement from blank project:
1. Define state/input model.
2. Implement linearization.
3. Build QP matrices for cost.
4. Add dynamics + input constraints.
5. Add corridor constraints.
6. Solve with OSQP.
7. Run receding horizon loop.
8. Add reference and speed scheduling.

## 11. Slide Deck Structure (Recommended)
1. Motivation and use-cases.
2. Theory recap (brief).
3. Code architecture.
4. QP mapping (equation -> matrix block).
5. Runtime loop and data flow.
6. Demo modes in UI.
7. Debugging and tuning.
8. Lab assignment and extensions.

## 12. Extensions for Advanced Students
- Add obstacle constraints.
- Add terminal cost/constraint.
- Compare EKF state vs true state in controller.
- Replace LTV linearization with nonlinear MPC baseline for comparison.
