import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from PIL import Image
import io
import os
from config.params import *  # Import HORIZON and REFERENCE_SPEED

def plot(states, refs, center=None, width=None, obstacles=None, prediction_horizons=None, computed_trajectories=None, save_path="mpc_trajectory.gif"):
    """
    Enhanced plotting with reference and actual trajectory comparison, prediction horizon visualization
    
    Args:
        states: Vehicle trajectory states
        refs: Reference trajectory points
        center: Road centerline points
        width: Road width
        obstacles: List of obstacle dictionaries
        prediction_horizons: List of prediction horizons for each step
        computed_trajectories: List of computed trajectories for each step
        save_path: Path to save the GIF file
    """
    s = np.array(states)
    r = np.array(refs)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot road boundaries if available
    if center is not None and width is not None:
        # Calculate road boundaries
        road_left = []
        road_right = []
        
        for i in range(len(center)):
            if i < len(center) - 1:
                tangent = center[i+1] - center[i]
            else:
                tangent = center[i] - center[i-1]
            
            tangent = tangent / np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            
            road_left.append(center[i] + normal * width/2)
            road_right.append(center[i] - normal * width/2)
        
        road_left = np.array(road_left)
        road_right = np.array(road_right)
        
        # Plot road surface
        ax.fill_between(road_left[:, 0], road_left[:, 1], road_right[:, 1], 
                       alpha=0.2, color='gray', label='Road')
        ax.plot(road_left[:, 0], road_left[:, 1], 'k-', linewidth=2, alpha=0.5)
        ax.plot(road_right[:, 0], road_right[:, 1], 'k-', linewidth=2, alpha=0.5)
        
        # Plot centerline
        ax.plot(center[:, 0], center[:, 1], 'k--', alpha=0.3, linewidth=1, label='Centerline')
        
        # Plot lane boundaries
        lane_left = []
        lane_right = []
        for i in range(len(center)):
            if i < len(center) - 1:
                tangent = center[i+1] - center[i]
            else:
                tangent = center[i] - center[i-1]
            
            tangent = tangent / np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            
            lane_left.append(center[i] + normal * width/4)
            lane_right.append(center[i] - normal * width/4)
        
        lane_left = np.array(lane_left)
        lane_right = np.array(lane_right)
        
        ax.plot(lane_left[:, 0], lane_left[:, 1], 'y--', alpha=0.3, linewidth=1, label='Lane boundary')
        ax.plot(lane_right[:, 0], lane_right[:, 1], 'y--', alpha=0.3, linewidth=1)
    
    # Plot obstacles (but remove main vehicle obstacle visualization)
    if obstacles is not None:
        for i, obs in enumerate(obstacles):
            # Handle both dictionary and DynamicObstacle objects
            if hasattr(obs, 'center'):
                center_obs = obs.center
                radius = obs.radius
                is_vehicle = hasattr(obs, 'speed')
            else:
                center_obs = obs['center']
                radius = obs['radius']
                is_vehicle = False
            
            # Draw as circle for all obstacles (no vehicle rectangles)
            circle = plt.Circle(center_obs, radius, 
                               color='red', alpha=0.7, label='Obstacle' if i == 0 else '')
            ax.add_patch(circle)
            
            # Safety margin (danger zone)
            danger_circle = plt.Circle(center_obs, radius + 1.0, 
                                      color='orange', alpha=0.2, linestyle='--', fill=False)
            ax.add_patch(danger_circle)
    
    # Plot reference trajectory from start to end (full trajectory)
    if len(r) > 0:
        ax.plot(r[:, 0], r[:, 1], 'g--', linewidth=2.5, alpha=0.8, label='Reference Trajectory')
        
        # Add reference trajectory markers
        marker_interval = max(1, len(r) // 20)  # Show markers at intervals
        for i in range(0, len(r), marker_interval):
            ax.plot(r[i, 0], r[i, 1], 'go', markersize=4, alpha=0.6)
    
    # Plot actual vehicle trajectory with distinct color and style
    if len(s) > 1:
        # Color gradient for actual trajectory
        colors = plt.cm.viridis(np.linspace(0, 1, len(s)))
        for i in range(len(s)-1):
            ax.plot(s[i:i+2, 0], s[i:i+2, 1], color=colors[i], linewidth=2.5, alpha=0.9)
        
        # Add actual trajectory markers
        marker_interval = max(1, len(s) // 20)  # Show markers at intervals
        for i in range(0, len(s), marker_interval):
            ax.plot(s[i, 0], s[i, 1], 'bo', markersize=4, alpha=0.7)
        
        # Start and end markers with colored dots
        ax.plot(s[0, 0], s[0, 1], 'go', markersize=15, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(s[-1, 0], s[-1, 1], 'ro', markersize=15, label='End', markeredgecolor='darkred', markeredgewidth=2)
        
        # Vehicle heading indicators at key points
        key_points = [0, len(s)//4, len(s)//2, 3*len(s)//4, len(s)-1]
        for idx in key_points:
            if idx < len(s):
                x, y, psi = s[idx, 0], s[idx, 1], s[idx, 2]
                # Arrow showing vehicle heading
                arrow_length = 1.5
                dx = arrow_length * np.cos(psi)
                dy = arrow_length * np.sin(psi)
                ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.2, 
                        fc='blue', ec='blue', alpha=0.7)
    
    # Plot prediction horizons (show every 10th step to avoid clutter) - FIXED
    if prediction_horizons is not None and len(prediction_horizons) > 0:
        horizon_interval = max(1, len(prediction_horizons) // 10)
        for i in range(0, len(prediction_horizons), horizon_interval):
            horizon = prediction_horizons[i]
            if len(horizon) > 0:
                # Plot prediction horizon as dotted line with consistent red color
                horizon_points = np.array([[p[0], p[1]] for p in horizon])
                # Use consistent red color for all prediction horizons
                ax.plot(horizon_points[:, 0], horizon_points[:, 1], 'r--', 
                       alpha=0.6, linewidth=1.5, 
                       label=f'Prediction Horizon (step {i})' if i == 0 else '')
                
                # Mark prediction horizon points
                ax.plot(horizon_points[:, 0], horizon_points[:, 1], 'r.', markersize=2, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('X [m]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=14, fontweight='bold')
    ax.set_title('MPC Trajectory Tracking: Reference vs Actual with Prediction Horizon', fontsize=16, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
    
    # Add text annotations
    if len(s) > 0 and len(r) > 0:
        total_distance = np.sum(np.sqrt(np.diff(s[:, 0])**2 + np.diff(s[:, 1])**2))
        avg_speed = np.mean(s[:, 3]) if len(s[0]) > 3 else 0
        
        # Calculate tracking error
        if len(s) == len(r):
            tracking_error = np.mean(np.linalg.norm(s[:, :2] - r[:, :2], axis=1))
        else:
            tracking_error = np.mean([np.linalg.norm(s[i, :2] - r[i, :2]) for i in range(min(len(s), len(r)))])
        
        info_text = f'Total Distance: {total_distance:.1f}m\nAvg Speed: {avg_speed:.1f}m/s\nTracking Error: {tracking_error:.2f}m\nTrajectory Points: {len(s)}\nPrediction Horizon: {HORIZON} steps ({HORIZON*DT:.1f}s)\nReference Speed: {REFERENCE_SPEED} m/s'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
               verticalalignment='top', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save as GIF file instead of showing
    plt.savefig(save_path.replace('.gif', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory plot saved as {save_path.replace('.gif', '.png')}")

def create_trajectory_gif(states, refs, center=None, width=None, obstacles=None, 
                         prediction_horizons=None, save_path="mpc_trajectory.gif", 
                         interval=50, fps=20):
    """
    Create animated GIF showing trajectory evolution with prediction horizons
    
    Args:
        states: Vehicle trajectory states
        refs: Reference trajectory points
        center: Road centerline points
        width: Road width
        obstacles: List of obstacle dictionaries
        prediction_horizons: List of prediction horizons for each step
        save_path: Path to save the GIF file
        interval: Interval between frames in milliseconds
        fps: Frames per second for the GIF
    """
    s = np.array(states)
    r = np.array(refs)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    def animate_frame(frame_idx):
        ax.clear()
        
        # Current frame data
        current_states = s[:frame_idx+1] if frame_idx < len(s) else s
        
        # Plot road boundaries
        if center is not None and width is not None:
            road_left = []
            road_right = []
            
            for i in range(len(center)):
                if i < len(center) - 1:
                    tangent = center[i+1] - center[i]
                else:
                    tangent = center[i] - center[i-1]
                
                tangent = tangent / np.linalg.norm(tangent)
                normal = np.array([-tangent[1], tangent[0]])
                
                road_left.append(center[i] + normal * width/2)
                road_right.append(center[i] - normal * width/2)
            
            road_left = np.array(road_left)
            road_right = np.array(road_right)
            
            ax.fill_between(road_left[:, 0], road_left[:, 1], road_right[:, 1], 
                           alpha=0.2, color='gray')
            ax.plot(road_left[:, 0], road_left[:, 1], 'k-', linewidth=2, alpha=0.5)
            ax.plot(road_right[:, 0], road_right[:, 1], 'k-', linewidth=2, alpha=0.5)
            ax.plot(center[:, 0], center[:, 1], 'k--', alpha=0.3, linewidth=1)
        
        # Plot reference trajectory from start to end (full trajectory, always visible)
        if len(r) > 0:
            ax.plot(r[:, 0], r[:, 1], 'g--', linewidth=2, alpha=0.8, label='Reference Trajectory')
        
        # Plot actual trajectory up to current frame
        if len(current_states) > 1:
            ax.plot(current_states[:, 0], current_states[:, 1], 'b-', linewidth=2.5, alpha=0.9, label='Actual Trajectory')
            
            # Current vehicle position
            current_x, current_y, current_psi = current_states[-1, 0], current_states[-1, 1], current_states[-1, 2]
            ax.plot(current_x, current_y, 'ro', markersize=8, label='Current Position')
            
            # Vehicle heading
            arrow_length = 1.5
            dx = arrow_length * np.cos(current_psi)
            dy = arrow_length * np.sin(current_psi)
            ax.arrow(current_x, current_y, dx, dy, head_width=0.3, head_length=0.2, 
                    fc='red', ec='red', alpha=0.7)
        
        # Plot obstacles (circles only, no vehicle rectangles)
        if obstacles is not None:
            for i, obs in enumerate(obstacles):
                if hasattr(obs, 'center'):
                    center_obs = obs.center
                    radius = obs.radius
                else:
                    center_obs = obs['center']
                    radius = obs['radius']
                
                circle = plt.Circle(center_obs, radius, color='red', alpha=0.7)
                ax.add_patch(circle)
        
        # Plot prediction horizon for current frame
        if prediction_horizons is not None and frame_idx < len(prediction_horizons):
            horizon = prediction_horizons[frame_idx]
            if len(horizon) > 0:
                horizon_points = np.array([[p[0], p[1]] for p in horizon])
                ax.plot(horizon_points[:, 0], horizon_points[:, 1], 'r--', 
                       linewidth=2, alpha=0.7, label='Prediction Horizon')
                
                # Mark prediction horizon points
                ax.plot(horizon_points[:, 0], horizon_points[:, 1], 'r.', markersize=3, alpha=0.5)
        
        # Add start and end markers (always visible)
        if len(s) > 0:
            ax.plot(s[0, 0], s[0, 1], 'go', markersize=12, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
            ax.plot(s[-1, 0], s[-1, 1], 'ro', markersize=12, label='End', markeredgecolor='darkred', markeredgewidth=2)
        
        # Formatting
        ax.set_xlabel('X [m]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y [m]', fontsize=12, fontweight='bold')
        ax.set_title(f'MPC Trajectory Tracking - Frame {frame_idx+1}/{len(s)}', fontsize=14, fontweight='bold')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add progress info
        if len(current_states) > 0:
            progress = (frame_idx + 1) / len(s) * 100
            speed = current_states[-1, 3] if len(current_states[-1]) > 3 else 0
            info_text = f'Progress: {progress:.1f}%\nSpeed: {speed:.1f} m/s'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   verticalalignment='top', fontsize=10, fontweight='bold')
    
    # Create animation
    print(f"Creating animated GIF with {len(s)} frames...")
    anim = animation.FuncAnimation(fig, animate_frame, frames=len(s), 
                                 interval=interval, repeat=True, blit=False)
    
    # Save as GIF
    print(f"Saving animation to {save_path}...")
    anim.save(save_path, writer='pillow', fps=fps, dpi=100)
    plt.close()
    
    print(f"Animated GIF saved as {save_path}")
    return anim
