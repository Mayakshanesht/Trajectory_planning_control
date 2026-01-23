"""
Gradio MPC Web Interface with Clickable Waypoints
------------------------------------------------
• Click to place exactly 4 waypoints (START → MID1 → MID2 → END)
• Cubic polynomial trajectory
• Real MPC tracking with track constraints
• Animated GIF output
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

from config.params import *
from track.track import sinusoidal_track, corridor_constraints
from utils.trajectory_utils import direct_waypoint_interpolation
from utils.plotting import create_trajectory_gif
from track.reference_trajectories import (
    zigzag_reference_trajectory,
    s_curve_reference_trajectory,
    sinusoidal_reference_trajectory_with_lane_changes
)

from vehicle.dynamics import step
from vehicle.actuators import ActuatorModel
from vehicle.sensors import measure
from estimation.ekf import EKF
from control.linearize import linearize
from control.mpc import MPC


# ---------------------------------------------------------
# Track (global, fixed)
# ---------------------------------------------------------
CENTER, WIDTH = sinusoidal_track(
    length=60.0,
    amplitude=5.0,
    width=7.4,
    points=250
)


# ---------------------------------------------------------
# Plot track + waypoints
# ---------------------------------------------------------
def plot_track_with_waypoints(waypoints):
    fig, ax = plt.subplots(figsize=(10, 4))

    # Centerline
    ax.plot(CENTER[:, 0], CENTER[:, 1], "--", color="gray", linewidth=1)

    # Road boundaries
    left, right = [], []
    for i in range(len(CENTER)):
        if i < len(CENTER) - 1:
            t = CENTER[i + 1] - CENTER[i]
        else:
            t = CENTER[i] - CENTER[i - 1]
        t = t / np.linalg.norm(t)
        n = np.array([-t[1], t[0]])
        left.append(CENTER[i] + n * WIDTH / 2)
        right.append(CENTER[i] - n * WIDTH / 2)

    left, right = np.array(left), np.array(right)
    ax.plot(left[:, 0], left[:, 1], color="black")
    ax.plot(right[:, 0], right[:, 1], color="black")

    # Waypoints
    labels = ["START", "MID 1", "MID 2", "END"]
    for i, (x, y) in enumerate(waypoints):
        ax.scatter(x, y, s=80, color="red", zorder=5)
        ax.text(x, y + 0.4, labels[i], ha="center", fontsize=9)

    ax.set_xlim(0, 60)
    ax.set_ylim(-8, 8)
    ax.set_aspect("equal")
    ax.set_title("Click on the road to place 4 waypoints")
    ax.grid(True)

    # Convert figure to numpy array for Gradio
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    img_array = img_array[:, :, :3]  # Remove alpha channel
    plt.close(fig)
    
    return img_array


# ---------------------------------------------------------
# Click handler
# ---------------------------------------------------------
def add_waypoint(waypoints, x, y):
    print(f"DEBUG: add_waypoint called with waypoints={waypoints}, x={x}, y={y}")
    
    if waypoints is None:
        waypoints = []
        print("DEBUG: waypoints was None, initialized to empty list")

    print(f"DEBUG: current waypoints count: {len(waypoints)}")
    
    if len(waypoints) >= 4:
        print("DEBUG: already have 4 waypoints, returning")
        return plot_track_with_waypoints(waypoints), waypoints

    print(f"DEBUG: adding waypoint: [{float(x)}, {float(y)}]")
    waypoints.append([float(x), float(y)])
    print(f"DEBUG: waypoints after adding: {waypoints}")
    
    try:
        result = plot_track_with_waypoints(waypoints)
        print("DEBUG: plot_track_with_waypoints succeeded")
        return result, waypoints
    except Exception as e:
        print(f"DEBUG: plot_track_with_waypoints failed: {e}")
        return plot_track_with_waypoints([]), waypoints


def clear_waypoints():
    return plot_track_with_waypoints([]), []


# ---------------------------------------------------------
# Reference trajectory from waypoints
# ---------------------------------------------------------
def build_reference(waypoints, speed):
    traj = direct_waypoint_interpolation(waypoints, N=120)
    traj = np.array(traj)

    dx = np.gradient(traj[:, 0])
    dy = np.gradient(traj[:, 1])
    psi = np.unwrap(np.arctan2(dy, dx))

    ref = np.zeros((len(traj), 6))
    ref[:, 0] = traj[:, 0]
    ref[:, 1] = traj[:, 1]
    ref[:, 2] = psi
    ref[:, 3] = speed

    return ref


# ---------------------------------------------------------
# MPC simulation (correct + stable)
# ---------------------------------------------------------
def run_mpc_simulation(waypoints, speed, horizon):

    ref_traj = build_reference(waypoints, speed)

    x = np.array([
        waypoints[0][0],
        waypoints[0][1],
        ref_traj[0, 2],
        speed,
        0.0,
        0.0
    ])
    u_prev = np.zeros(2)

    act = ActuatorModel(
        LIMITS["steer_rate"],
        LIMITS["steer_max"],
        LIMITS["accel_max"]
    )
    ekf = EKF(DT, VEHICLE)

    Q = [120, 120, 30, 15, 10, 10]
    R = [12.0, 5.0]
    Rj = [60.0, 20.0]

    mpc = MPC(
        horizon,
        Q, R, Rj,
        [-LIMITS["steer_max"], -LIMITS["accel_max"]],
        [ LIMITS["steer_max"],  LIMITS["accel_max"]]
    )

    states = []
    errors = []
    last_ref_idx = 0

    # Continue simulation until we reach the end of the reference trajectory
    while True:
        dists = np.linalg.norm(ref_traj[:, :2] - x[:2], axis=1)
        idx = max(np.argmin(dists), last_ref_idx)
        last_ref_idx = idx

        # Break if we've reached the end of the reference trajectory
        if idx >= len(ref_traj) - horizon - 1:
            print(f"DEBUG: Reached end of reference trajectory at idx {idx}")
            break
            
        ref = ref_traj[idx:idx + horizon]

        z = measure(x, NOISE)
        ekf.predict(u_prev)
        xhat = ekf.update(z)

        A, B, c, track_cons = [], [], [], []
        for k in range(horizon):
            a, b, cc = linearize(ref[k], u_prev, DT, VEHICLE)
            A.append(a)
            B.append(b)
            c.append(cc)

            ci = min(idx + k, len(CENTER) - 1)
            track_cons.append(corridor_constraints(CENTER, WIDTH, ci))

        u = mpc.solve(xhat, ref, A, B, c, track_cons, [])
        u = u[:2]  # Take only the first control input (steering, acceleration)
        u = act.apply(u, DT)

        x = step(x, u, DT, VEHICLE)
        u_prev = u

        states.append(x.copy())
        errors.append(np.linalg.norm(x[:2] - ref[0, :2]))

    states = np.array(states)

    # Use the actual reference point the vehicle was following at the end
    # This is more accurate than just taking the last point of the generated trajectory
    final_ref_idx = last_ref_idx  # This is the index of the reference point we were actually following
    actual_endpoint = ref_traj[final_ref_idx, :2]  # Use the reference point we were actually following
    
    print(f"DEBUG: Manual waypoints - Final ref idx: {final_ref_idx}")
    print(f"DEBUG: Manual waypoints - Using actual ref point: {actual_endpoint}")
    print(f"DEBUG: Manual waypoints - Last manual waypoint was: {waypoints[-1]}")
    print(f"DEBUG: Manual waypoints - Generated trajectory endpoint: {ref_traj[-1, :2]}")

    metrics = {
        "final_error": float(np.linalg.norm(states[-1, :2] - actual_endpoint)),
        "avg_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "path_length": float(np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1))),
        "steps": len(states),
    }

    gif_path = "mpc_gradio.gif"
    
    # Interpolate the reference trajectory to match the actual trajectory length
    if len(states) > 1 and len(ref_traj) > 1:
        # Create interpolation indices for the reference trajectory
        ref_indices = np.linspace(0, len(ref_traj) - 1, len(states))
        ref_indices = np.clip(ref_indices, 0, len(ref_traj) - 1).astype(int)
        
        # Interpolate the reference trajectory
        ref_interp = ref_traj[ref_indices]
        
        print(f"DEBUG: Interpolated reference trajectory from {len(ref_traj)} to {len(states)} points")
    else:
        ref_interp = ref_traj[:len(states)] if len(states) <= len(ref_traj) else ref_traj
    
    create_trajectory_gif(
        states,
        ref_interp,
        CENTER,
        WIDTH,
        [],
        save_path=gif_path,
        fps=20
    )

    return metrics, gif_path


def test_ref_trajectory(traj_type, speed, horizon):
    """Simple test function to verify button clicks work"""
    print(f"TEST: Reference trajectory button clicked!")
    print(f"TEST: traj_type={traj_type}, speed={speed}, horizon={horizon}")
    
    # Create a simple test without MPC
    try:
        center, width = sinusoidal_track(length=60.0, amplitude=5.0, width=7.4, points=250)
        
        report = f"""
✅ SIMPLE TEST SUCCESSFUL

• Trajectory Type: {traj_type}
• Speed: {speed} m/s
• Horizon: {horizon} steps
• Track Length: {len(center)} points
• Track Width: {width} m
• Status: Basic functionality working!

This is a simple test without MPC simulation.
"""
        
        return report, None
        
    except Exception as e:
        error_report = f"❌ Simple test failed: {str(e)}"
        return error_report, None


def run_reference_trajectory(traj_type, speed, horizon):
    """Run MPC simulation with reference trajectory"""
    print(f"DEBUG: run_reference_trajectory called with traj_type={traj_type}, speed={speed}, horizon={horizon}")
    
    try:
        # Create track
        print("DEBUG: Creating track...")
        center, width = sinusoidal_track(length=60.0, amplitude=5.0, width=7.4, points=250)
        print(f"DEBUG: Track created with {len(center)} points")
        
        # Initialize vehicle
        print("DEBUG: Initializing vehicle...")
        start_position = center[0].copy()
        start_heading = 0.0
        x = np.array([start_position[0], start_position[1], start_heading, speed, 0.0, 0.0])
        u_prev = np.zeros(2)
        print(f"DEBUG: Vehicle initialized at position {start_position}")
        
        # Initialize components
        print("DEBUG: Initializing MPC components...")
        act = ActuatorModel(LIMITS["steer_rate"], LIMITS["steer_max"], LIMITS["accel_max"])
        ekf = EKF(DT, VEHICLE)
        
        # MPC weights
        Q_TRACK_SAFE = [120, 120, 30, 15, 10, 10]
        R_INPUT_SAFE = [12.0, 5.0]
        R_JERK_SAFE = [60.0, 20.0]
        
        mpc = MPC(horizon, Q_TRACK_SAFE, R_INPUT_SAFE, R_JERK_SAFE,
                  [-LIMITS["steer_max"], -LIMITS["accel_max"]],
                  [ LIMITS["steer_max"],  LIMITS["accel_max"]])
        print("DEBUG: MPC components initialized")
        
        # Run simulation
        print("DEBUG: Starting simulation loop...")
        states = [x.copy()]
        errors = []
        final_ref_point = center[-1].copy()  # Initialize with default endpoint
        
        for step_idx in range(200):  # Run for 200 steps
            if step_idx % 50 == 0:
                print(f"DEBUG: Step {step_idx}/200")
            
            # Generate reference trajectory based on type
            current_progress_idx = int(step_idx * len(center) / 200)
            current_progress = current_progress_idx / len(center)
            
            print(f"DEBUG: Generating {traj_type} trajectory at progress {current_progress:.2f}")
            
            if traj_type == "ref1":
                # Sinusoidal with lane changes - use the same logic as main_simple.py
                lane_width = width / 2
                left_lane_offset = -lane_width / 2
                right_lane_offset = lane_width / 2
                
                # Simple lane change logic
                if current_progress < 0.3:
                    current_target_offset = left_lane_offset
                elif current_progress < 0.7:
                    current_target_offset = right_lane_offset
                else:
                    current_target_offset = left_lane_offset
                
                print(f"DEBUG: Using lane offset: {current_target_offset}")
                
                ref = sinusoidal_reference_trajectory_with_lane_changes(
                    x, center, horizon, DT, speed=speed,
                    target_lane_offset=current_target_offset, lane_width=lane_width,
                    current_idx=current_progress_idx
                )
                print(f"DEBUG: Generated ref trajectory with shape: {ref.shape}")
                
                # Calculate proper endpoint based on final lane position
                if current_progress >= 0.7:
                    final_ref_point = np.array([center[-1, 0], center[-1, 1] + left_lane_offset])
                else:
                    final_ref_point = np.array([center[-1, 0], center[-1, 1] + right_lane_offset])
                    
            elif traj_type == "ref2":
                # Zigzag pattern
                print("DEBUG: Generating zigzag trajectory...")
                ref = zigzag_reference_trajectory(
                    x, center, horizon, DT, speed=speed,
                    amplitude_offset=2.0, current_idx=current_progress_idx
                )
                print(f"DEBUG: Generated ref trajectory with shape: {ref.shape}")
                # For zigzag, the final position should be near the track center
                final_ref_point = center[-1].copy()
                
            elif traj_type == "ref3":
                # S-curve pattern
                print("DEBUG: Generating S-curve trajectory...")
                ref = s_curve_reference_trajectory(
                    x, center, horizon, DT, speed=speed,
                    current_idx=current_progress_idx
                )
                print(f"DEBUG: Generated ref trajectory with shape: {ref.shape}")
                # For S-curve, the final position should be near the track center
                final_ref_point = center[-1].copy()
                
            else:
                print(f"DEBUG: Unknown trajectory type: {traj_type}")
                return f"❌ Unknown trajectory type: {traj_type}", None
            
            print(f"DEBUG: Building MPC matrices for step {step_idx}")
            # Build MPC matrices
            A, B, c = [], [], []
            track_cons = []
            
            for k in range(horizon):
                a, b, cc = linearize(ref[k], u_prev, DT, VEHICLE)
                A.append(a); B.append(b); c.append(cc)
                
                ci = min(current_progress_idx + k, len(center) - 1)
                track_cons.append(corridor_constraints(center, width, ci))
            
            print(f"DEBUG: Solving MPC for step {step_idx}")
            # Solve MPC
            u = mpc.solve(x, ref, A, B, c, track_cons, [])
            u = u[:2]  # Take only first control input
            u = act.apply(u, DT)
            
            # Update vehicle
            x = step(x, u, DT, VEHICLE)
            u_prev = u
            
            states.append(x.copy())
            errors.append(np.linalg.norm(x[:2] - ref[0, :2]))
        
        print("DEBUG: Simulation completed, calculating metrics...")
        
        # Calculate metrics - final_ref_point was calculated in the loop
        states = np.array(states)
        
        print(f"DEBUG: Using trajectory endpoint: {final_ref_point}")
        
        metrics = {
            "final_error": float(np.linalg.norm(states[-1, :2] - final_ref_point)),
            "avg_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "path_length": float(np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1))),
            "steps": len(states),
        }
        
        print("DEBUG: Creating GIF...")
        # Create GIF
        gif_path = create_trajectory_gif(states, [], [], save_path=f'mpc_{traj_type}_tracking.gif')
        
        report = f"""
📊 {traj_type.upper()} MPC RESULTS

• Final error: {metrics['final_error']:.2f} m
• Average error: {metrics['avg_error']:.2f} m
• Max error: {metrics['max_error']:.2f} m
• Path length: {metrics['path_length']:.2f} m
• Simulation steps: {metrics['steps']}
• Trajectory type: {traj_type}
• Final position: ({states[-1, 0]:.2f}, {states[-1, 1]:.2f}) m
• Target position: ({final_ref_point[0]:.2f}, {final_ref_point[1]:.2f}) m
"""
        
        print(f"DEBUG: {traj_type} simulation completed successfully")
        return report, gif_path
        
    except Exception as e:
        print(f"DEBUG: Error in {traj_type} trajectory: {e}")
        import traceback
        traceback.print_exc()
        error_report = f"❌ Error running {traj_type} trajectory: {str(e)}"
        return error_report, None


# ---------------------------------------------------------
# Reference trajectory callback
# ---------------------------------------------------------
def run_from_ui(waypoints, speed, horizon):

    if waypoints is None or len(waypoints) != 4:
        return "❌ Please click exactly 4 waypoints.", None

    metrics, gif = run_mpc_simulation(waypoints, speed, horizon)
    
    # Get the reference trajectory for reporting
    ref_traj = build_reference(waypoints, speed)
    
    report = f"""
📊 MANUAL WAYPOINTS MPC RESULTS

• Final error: {metrics['final_error']:.2f} m
• Average error: {metrics['avg_error']:.2f} m
• Max error: {metrics['max_error']:.2f} m
• Path length: {metrics['path_length']:.2f} m
• Simulation steps: {metrics['steps']}
• Waypoints used: {[f"P{i+1}({w[0]:.1f},{w[1]:.1f})" for i,w in enumerate(waypoints)]}
• Generated trajectory points: {len(ref_traj)}
• Status: Following smooth cubic polynomial through waypoints
"""

    return report, gif


# ---------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------
demo = gr.Blocks(title="MPC Trajectory Tracking")

with demo:

    gr.Markdown("## 🚗 MPC Trajectory Tracking with Manual Waypoint Entry")
    gr.Markdown("Choose between **manual waypoints** or **reference trajectories**")
    
    with gr.Tabs():
        with gr.TabItem("Manual Waypoints"):
            gr.Markdown("Enter **exactly 4 waypoints** manually: START → MID1 → MID2 → END")
            gr.Markdown("Format: x,y (e.g., 5.0,1.2). Valid range: x=[0,60], y=[-8,8]")

            waypoint_state = gr.State([])

            with gr.Row():
                x_input = gr.Number(label="X coordinate", minimum=0, maximum=60, value=10)
                y_input = gr.Number(label="Y coordinate", minimum=-8, maximum=8, value=0)
                add_btn = gr.Button("Add Waypoint")

            track_plot = gr.Image(
                value=plot_track_with_waypoints([]),
                label="Track Visualization"
            )

            add_btn.click(
                add_waypoint,
                inputs=[waypoint_state, x_input, y_input],
                outputs=[track_plot, waypoint_state]
            )

            with gr.Row():
                clear_btn = gr.Button("Clear Waypoints")
                run_btn = gr.Button("Run MPC")

        with gr.TabItem("Reference Trajectories"):
            gr.Markdown("Choose from predefined reference trajectories")
            
            traj_type = gr.Dropdown(
                choices=[
                    ("Sinusoidal with Lane Changes", "ref1"),
                    ("Zigzag Pattern", "ref2"), 
                    ("S-Curve Pattern", "ref3")
                ],
                value="ref1",
                label="Reference Trajectory Type"
            )
            
            ref_run_btn = gr.Button("Run Reference Trajectory")
            ref_track_plot = gr.Image(
                value=plot_track_with_waypoints([]),
                label="Reference Track Visualization"
            )

    speed = gr.Slider(1, 10, value=4.0, step=0.1, label="Reference speed (m/s)")
    horizon = gr.Slider(15, 60, value=25, step=5, label="MPC horizon")

    output_text = gr.Textbox(lines=10, label="Results")
    output_gif = gr.Image(type="filepath", label="MPC Animation")

    ref_run_btn.click(
        run_reference_trajectory,
        inputs=[traj_type, speed, horizon],
        outputs=[output_text, output_gif]
    )

    clear_btn.click(
        clear_waypoints,
        outputs=[track_plot, waypoint_state]
    )

    run_btn.click(
        run_from_ui,
        inputs=[waypoint_state, speed, horizon],
        outputs=[output_text, output_gif]
    )


# ---------------------------------------------------------
# Launch
# ---------------------------------------------------------
if __name__ == "__main__":
    demo.launch()
