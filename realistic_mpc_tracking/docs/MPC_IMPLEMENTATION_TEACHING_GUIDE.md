# MPC Implementation Teaching Guide (Theory -> Code)

Audience: students who already know MPC theory and now need to implement it correctly.

This guide is written so you can teach directly from it or convert each section into slides.

## 1. Learning Outcomes
By the end, students should be able to:
1. Translate MPC equations into QP matrices (`P, q, A, l, u`).
2. Explain how linearization produces LTV MPC.
3. Implement and debug a receding-horizon control loop.
4. Tune MPC weights and horizon using measurable outcomes.

## 2. Prerequisites (Keep Explicit)
Students should already know:
1. Linear algebra (matrix multiplication, block matrices).
2. Constrained optimization basics.
3. State-space models and linearization.
4. Python + NumPy fundamentals.

## 3. Big Picture Pipeline (Slide: One Diagram)
Use this as your core architecture slide:

1. Build reference path.
2. Convert path into reference states `[x, y, psi, v, 0, 0]`.
3. Estimate current state with EKF from noisy measurements.
4. Linearize dynamics at each horizon step: `(A_k, B_k, c_k)`.
5. Build QP objective and constraints.
6. Solve QP with OSQP.
7. Apply first control only.
8. Advance simulation and repeat.

Code anchors:
1. `web_interface.py` (runtime loop + reference generation)
2. `control/linearize.py` (linearization)
3. `control/mpc.py` (QP build + solve)
4. `vehicle/dynamics.py` (state propagation)
5. `track/track.py` (road corridor constraints)

## 3.1 Repository Map (What Each File/Folder Does)
Use this section to orient students before diving into equations.

Top-level scripts:
1. `launcher.py`: single entrypoint that launches web/cli flows.
2. `web_interface.py`: main teaching app; builds references, runs MPC loop, reports metrics/GIF.
3. `main_simple.py`: CLI-only simulation runner for predefined trajectory experiments.
4. `requirements.txt`: pinned dependencies for reproducible environment setup.
5. `README.md`: user setup and workflow documentation.

`config/`:
1. `config/params.py`: global parameters (dt, limits, horizon defaults, noise, etc.).

`control/`:
1. `control/linearize.py`: computes local linear model `(A_k, B_k, c_k)` for MPC.
2. `control/mpc.py`: assembles QP matrices (`P,q,A,l,u`) and solves with OSQP.

`vehicle/`:
1. `vehicle/dynamics.py`: vehicle state update for one timestep.
2. `vehicle/actuators.py`: actuator saturation/rate limiting model.
3. `vehicle/sensors.py`: measurement model used by estimator.

`estimation/`:
1. `estimation/ekf.py`: EKF prediction/update used before optimization.

`track/`:
1. `track/track.py`: road geometry and corridor half-space constraints.
2. `track/reference_trajectories.py`: predefined reference trajectory generators.

`utils/`:
1. `utils/plotting.py`: visualization and animated GIF creation.
2. `utils/trajectory_utils.py`: helper functions for trajectory interpolation/parsing.

`docs/`:
1. `docs/MPC_IMPLEMENTATION_TEACHING_GUIDE.md`: teaching/slide-conversion guide.

## 3.2 How EKF Is Enacted Here (What, Where, Why)
What EKF is doing in this project:
1. Controller does not use raw noisy measurements directly.
2. It estimates full state `xhat` each timestep.
3. MPC optimization is solved around `xhat`, not raw `z`.

Where it runs in code:
1. `estimation/ekf.py`: EKF class implementation.
2. `web_interface.py`: in control loop, order is:
   - `z = measure(x, NOISE)`
   - `ekf.predict(u_prev)`
   - `xhat = ekf.update(z)`
3. `main_simple.py`: same predict/update pattern in CLI loop.

How this EKF is formed:
1. State dimension is 6: `[X, Y, psi, vx, vy, r]`.
2. Prediction model uses `vehicle/dynamics.py::step(...)`.
3. Covariance prediction uses simplified form `P = P + Q`.
4. Measurement model in `estimation/ekf.py` uses fixed `H` that observes:
   - yaw rate `r` (state index 5),
   - longitudinal speed `vx` (index 3),
   - position `X` (index 0),
   - position `Y` (index 1).
5. Update is standard EKF linear measurement update with Kalman gain `K`.

Why EKF is needed for teaching and implementation:
1. Real sensors are noisy/incomplete; raw signals destabilize MPC tuning.
2. Estimation separates "state reconstruction" from "control optimization".
3. Students learn practical architecture used in real systems:
   - estimator block -> controller block -> plant.
4. It explains why two states exist in code:
   - true simulated state `x`,
   - estimated control state `xhat`.

Slide placement recommendation:
1. Put EKF slide immediately after "Model and State Definition".
2. Then show the receding-horizon loop.
3. Then show QP matrix construction.

## 4. Equation -> Code Map (Most Important Teaching Slide)
Teach this mapping explicitly:

Objective:
- Theory:
  `min sum ||x_k - x_ref,k||_Q^2 + sum ||u_k||_R^2 + sum ||u_{k+1}-u_k||_{Rj}^2`
- Code:
  - `control/mpc.py::_build_cost(...)`
  - State block in `P` from `Q`
  - Input block in `P` from `R`
  - Jerk/smoothness block in `P` from `Rj`

Dynamics constraints:
- Theory:
  `x_{k+1} = A_k x_k + B_k u_k + c_k`
- Code:
  - `control/mpc.py::_build_dynamics_constraints(...)`
  - `control/linearize.py` supplies `A_k, B_k, c_k`

Input bounds:
- Theory:
  `u_min <= u_k <= u_max`
- Code:
  - `control/mpc.py::_build_input_constraints(...)`

Road corridor:
- Theory: half-space lane limits.
- Code:
  - `track/track.py::corridor_constraints(...)`
  - `control/mpc.py::_build_track_constraints(...)`

Solver:
- Theory: solve QP each timestep.
- Code:
  - `control/mpc.py::solve(...)` using OSQP.

## 5. Receding Horizon Loop (Teach as a Deterministic Algorithm)
Use this exact sequence in class:

1. Measure/estimate current state.
2. Select nearest forward progress index on reference.
3. Slice `N` reference points.
4. Linearize at each `k in [0, N-1]`.
5. Build QP matrices and bounds.
6. Solve for control sequence.
7. Apply only `u_0`.
8. Simulate one step.
9. Repeat until reference end or stop condition.

Teaching emphasis:
- Students often think MPC applies the full control sequence.
  Correct this early: only the first action is executed.

## 6. Why LTV Here (and Not LTI)
Simple classroom explanation:
1. Vehicle model is nonlinear.
2. We linearize around the current predicted trajectory.
3. So model matrices vary with time/index (`A_k`, `B_k`, `c_k`).
4. That is Linear Time-Varying MPC.

## 7. Reference Quality Matters More Than Most Students Expect
Current project supports:
1. Predefined references (`ref1`, `ref2`, `ref3`).
2. Upload file (`x,y` or `s,offset`).
3. Guided anchor offsets (START, +peak, -peak, END).

Implementation note:
1. Reference speed is curvature-aware (higher curvature -> lower reference speed).
2. This improves robustness for zigzag/lane-change paths.

Teaching point:
1. Bad reference design can make a good MPC look bad.

## 8. Failure Modes and Debug Decision Tree
When results are poor, use this order:

1. Is reference feasible inside road bounds?
2. Is reference speed too high for curvature?
3. Is horizon too short?
4. Are `Q/R/Rj` balanced?
5. Are actuator bounds too strict?
6. Is progress index logic monotonic forward?

Where to inspect:
1. `web_interface.py`: reference path + progress logic.
2. `control/mpc.py`: cost/constraints.
3. `control/linearize.py`: model validity.

## 9. Suggested 90-Minute Teaching Plan
1. 0-10 min: architecture and data flow.
2. 10-30 min: equation-to-code map in `control/mpc.py`.
3. 30-45 min: linearization and LTV interpretation.
4. 45-60 min: receding-horizon loop walkthrough.
5. 60-75 min: run demos in web UI.
6. 75-90 min: tuning lab + discussion.

## 10. Slide Deck Blueprint (Ready to Build)
Recommended slide sequence:
1. Problem statement and goals.
2. MPC equation recap (short).
3. Repo architecture.
4. State estimation block (EKF): what is measured vs estimated.
5. Cost function -> matrix `P,q`.
6. Constraints -> matrix `A,l,u`.
7. Linearization (`A_k,B_k,c_k`).
8. Receding horizon runtime loop.
9. Reference generation modes.
10. Debug checklist.
11. Tuning lab and expected trends.

## 11. Hands-On Labs (With Measurable Outputs)
Lab A: Horizon sensitivity
1. Fix reference type `ref2`.
2. Test `N = 20, 30, 40, 50`.
3. Record final/avg/max error and progress.

Lab B: Speed sensitivity
1. Fix horizon.
2. Sweep speed `2, 3, 4, 5 m/s`.
3. Compare tracking vs stability.

Lab C: Weight tuning
1. Increase `Q` position weights.
2. Increase `R` input weights.
3. Increase `Rj` smoothness weights.
4. Explain behavior changes in plots.

## 12. Assessment Rubric (Optional for Course Use)
Good implementation should show:
1. Correct matrix assembly and dimensions.
2. Stable closed-loop behavior on at least 2 reference types.
3. Clear explanation of chosen `Q/R/Rj`.
4. Reproducible experiment logs (horizon/speed sweeps).

## 13. Extension Topics (Advanced)
1. Add obstacle constraints in QP.
2. Add terminal cost and terminal set ideas.
3. Compare EKF-estimated vs true state feedback.
4. Compare LTV MPC with nonlinear MPC baseline.

## 14. Instructor Notes (Practical)
1. Start from one successful trajectory before showing hard ones.
2. Keep one "broken" configuration ready for debugging demos.
3. Ask students to predict behavior before each tuning run.
4. Grade reasoning quality, not only low tracking error.
