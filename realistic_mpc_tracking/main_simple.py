"""
Simplified MPC trajectory tracking with command line interface
"""
import numpy as np
import argparse
from config.params import *
from vehicle.dynamics import step
from vehicle.actuators import ActuatorModel
from vehicle.sensors import measure
from estimation.ekf import EKF
from control.linearize import linearize
from control.mpc import MPC
from track.track import sinusoidal_track, corridor_constraints
from track.reference_trajectories import (
    sinusoidal_reference_trajectory_with_lane_changes,
    zigzag_reference_trajectory,
    s_curve_reference_trajectory,
    polynomial_reference_trajectory
)
from utils.plotting import create_trajectory_gif

def run_simulation(trajectory_type="ref1", waypoints=None, custom_params=None):
    """
    Run MPC simulation with specified trajectory type
    
    Args:
        trajectory_type: "ref1", "ref2", "ref3", or "custom"
        waypoints: List of (x, y) points for custom trajectory
        custom_params: Dictionary with custom parameters
    """
    # Use custom parameters if provided, otherwise use defaults
    if custom_params:
        globals().update(custom_params)
    
    # Create sinusoidal track
    center, width = sinusoidal_track(length=60.0, amplitude=5.0, width=7.4, points=250)
    
    # Initialize vehicle
    start_position = center[0].copy()
    start_heading = 0.0
    x = np.array([start_position[0], start_position[1], start_heading, 3.0, 0.0, 0.0])
    u_prev = np.zeros(2)
    
    # Initialize components
    act = ActuatorModel(LIMITS["steer_rate"], LIMITS["steer_max"], LIMITS["accel_max"])
    ekf = EKF(DT, VEHICLE)
    
    # MPC weights
    Q_TRACK_SAFE = [120, 120, 30, 15, 10, 10]
    R_INPUT_SAFE = [12.0, 5.0]
    R_JERK_SAFE = [60.0, 20.0]
    
    mpc = MPC(HORIZON, Q_TRACK_SAFE, R_INPUT_SAFE, R_JERK_SAFE,
              [-LIMITS["steer_max"], -LIMITS["accel_max"]],
              [ LIMITS["steer_max"],  LIMITS["accel_max"]])
    
    states = []
    refs = []
    prediction_horizons = []
    
    print(f"Starting {trajectory_type} simulation...")
    print(f"Track length: 60m, Amplitude: 5m, Width: 7.4m")
    print(f"Reference speed: {REFERENCE_SPEED} m/s")
    
    for t in range(SIM_STEPS * 2):
        # Progress tracking
        current_progress_idx = np.argmin(np.linalg.norm(center - x[:2], axis=1))
        current_progress = current_progress_idx / len(center)
        
        # Check if reached destination
        if current_progress_idx >= len(center) - 5:
            print(f"🏁 Destination reached! Progress: {current_progress*100:.1f}%")
            
            # Stop vehicle
            if x[3] > 0.5:
                u = np.array([0.0, -3.0])
            elif x[3] > 0.1:
                u = np.array([0.0, -1.0])
            else:
                u = np.array([0.0, 0.0])
            
            x = step(x, u, DT, VEHICLE)
            states.append(x)
            refs.append(ref[0] if 'ref' in locals() else np.array([x[0], x[1], x[2], 0, 0, 0]))
            
            if x[3] <= 0.1:
                print(f"✅ Vehicle stopped at destination. Final position: ({x[0]:.1f}, {x[1]:.1f}), Final speed: {x[3]:.2f} m/s")
                break
        
        # Adaptive speed
        if current_progress_idx > len(center) - 20:
            distance_to_end = len(center) - current_progress_idx
            adaptive_speed = max(0.5, REFERENCE_SPEED * (distance_to_end / 20))
        else:
            adaptive_speed = REFERENCE_SPEED
        
        # State estimation
        z = measure(x, NOISE)
        ekf.predict(u_prev)
        xhat = ekf.update(z)
        
        # Generate reference trajectory
        if trajectory_type == "ref1":
            # Sinusoidal with lane changes
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
            
            ref = sinusoidal_reference_trajectory_with_lane_changes(
                xhat, center, HORIZON, DT, speed=adaptive_speed,
                target_lane_offset=current_target_offset, lane_width=lane_width,
                current_idx=current_progress_idx
            )
            
        elif trajectory_type == "ref2":
            # Zigzag pattern
            ref = zigzag_reference_trajectory(
                xhat, center, HORIZON, DT, speed=adaptive_speed,
                amplitude_offset=2.0, current_idx=current_progress_idx
            )
            
        elif trajectory_type == "ref3":
            # S-curve pattern
            ref = s_curve_reference_trajectory(
                xhat, center, HORIZON, DT, speed=adaptive_speed,
                current_idx=current_progress_idx
            )
            
        elif trajectory_type == "custom" and waypoints:
            # Custom polynomial trajectory
            ref = polynomial_reference_trajectory(
                xhat, center, waypoints, HORIZON, DT, speed=adaptive_speed,
                current_idx=current_progress_idx
            )
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
        
        # MPC optimization
        A, B, c = [], [], []
        track_cons = []
        
        for k in range(HORIZON):
            a, b, cc = linearize(ref[k], u_prev, DT, VEHICLE)
            A.append(a); B.append(b); c.append(cc)
            
            idx = np.argmin(np.linalg.norm(center - ref[k,:2], axis=1))
            track_cons.append(corridor_constraints(center, width, idx))
        
        u_cmd = mpc.solve(xhat, ref, A, B, c, track_cons, [])
        
        # Apply control
        u = u_cmd[:2]
        u = act.apply(u, DT)
        
        # Compute prediction horizon
        predicted_trajectory = []
        x_pred = xhat.copy()
        
        for k in range(HORIZON):
            if k * 2 < len(u_cmd):
                u_pred = u_cmd[k*2:k*2+2]
                u_pred = np.clip(u_pred, [-LIMITS["steer_max"], -LIMITS["accel_max"]], 
                                [LIMITS["steer_max"], LIMITS["accel_max"]])
            else:
                u_pred = np.zeros(2)
            
            predicted_trajectory.append(x_pred.copy())
            x_pred = step(x_pred, u_pred, DT, VEHICLE)
        
        prediction_horizons.append(np.array(predicted_trajectory))
        
        # Update vehicle state
        x = step(x, u, DT, VEHICLE)
        u_prev = u
        
        states.append(x)
        refs.append(ref[0])
        
        if t % 50 == 0:
            print(f"Step {t}/{SIM_STEPS*2}: Position ({x[0]:.1f}, {x[1]:.1f}), Speed: {x[3]:.1f}, Progress: {current_progress*100:.1f}%")
    
    # Create visualization
    print("\nSimulation complete! Creating animated GIF...")
    
    try:
        anim = create_trajectory_gif(
            states, refs, center, width, [], 
            prediction_horizons=prediction_horizons,
            save_path=f"mpc_{trajectory_type}_tracking.gif",
            interval=50, fps=20
        )
        print(f"Animated GIF saved as 'mpc_{trajectory_type}_tracking.gif'")
        return True
    except Exception as e:
        print(f"GIF creation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='MPC Trajectory Tracking Simulation')
    parser.add_argument('--trajectory', type=str, default='ref1',
                        choices=['ref1', 'ref2', 'ref3'],
                        help='Reference trajectory type')
    parser.add_argument('--speed', type=float, default=4.0,
                        help='Reference speed (m/s)')
    parser.add_argument('--horizon', type=int, default=60,
                        help='MPC prediction horizon')
    parser.add_argument('--output', type=str, default='mpc_trajectory.gif',
                        help='Output GIF filename')
    
    args = parser.parse_args()
    
    # Custom parameters
    custom_params = {
        'REFERENCE_SPEED': args.speed,
        'HORIZON': args.horizon
    }
    
    # Run simulation
    success = run_simulation(args.trajectory, custom_params=custom_params)
    
    if success:
        print(f"\n✅ Simulation completed successfully!")
        print(f"📁 Output: {args.output}")
    else:
        print(f"\n❌ Simulation failed!")

if __name__ == "__main__":
    main()
