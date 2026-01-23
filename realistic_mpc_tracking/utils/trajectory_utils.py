"""
Shared trajectory utilities for all interfaces
"""
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import *
from track.track import sinusoidal_track

# Global variables
center, width = sinusoidal_track(length=60.0, amplitude=5.0, width=7.4, points=250)

def direct_waypoint_interpolation(waypoints, N=100):
    """
    Direct interpolation through waypoints using cubic polynomials
    """
    waypoints = np.array(waypoints)
    
    # Parameterize the curve using arc length parameter t
    t_values = np.linspace(0, 1, N)
    
    # Fit x(t) and y(t) separately as cubic polynomials
    if len(waypoints) >= 4:
        # Cubic polynomial (3rd degree) for 4 points
        coeffs_x = np.polyfit(np.linspace(0, 1, len(waypoints)), waypoints[:, 0], 3)
        coeffs_y = np.polyfit(np.linspace(0, 1, len(waypoints)), waypoints[:, 1], 3)
    elif len(waypoints) == 3:
        # Quadratic polynomial (2nd degree) for 3 points
        coeffs_x = np.polyfit(np.linspace(0, 1, len(waypoints)), waypoints[:, 0], 2)
        coeffs_y = np.polyfit(np.linspace(0, 1, len(waypoints)), waypoints[:, 1], 2)
    else:
        # Linear interpolation for 2 points
        coeffs_x = np.polyfit(np.linspace(0, 1, len(waypoints)), waypoints[:, 0], 1)
        coeffs_y = np.polyfit(np.linspace(0, 1, len(waypoints)), waypoints[:, 1], 1)
    
    # Generate interpolated points
    trajectory = []
    for t in t_values:
        x = np.polyval(coeffs_x, t)
        y = np.polyval(coeffs_y, t)
        trajectory.append([x, y])
    
    return np.array(trajectory)

def create_track_plot_with_waypoints(waypoints_list, title="Click on the road to add waypoints"):
    """Create track plot with waypoints"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot road
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
    
    # Add start and end markers
    ax.plot(center[0, 0], center[0, 1], 'go', markersize=12, label='START')
    ax.plot(center[-1, 0], center[-1, 1], 'ro', markersize=12, label='END')
    
    # Plot existing waypoints
    if waypoints_list:
        points_array = np.array(waypoints_list)
        ax.plot(points_array[:, 0], points_array[:, 1], 'bo-', markersize=8, linewidth=2, label='Waypoints')
        
        # Add waypoint numbers
        for i, (x, y) in enumerate(waypoints_list):
            ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold', color='blue')
    
    ax.set_xlabel('X [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
    plt.close()
    
    return temp_file.name

def create_trajectory_plot(waypoints, trajectory, speed=4.0, horizon=60, title="Generated Trajectory"):
    """Create trajectory plot with waypoints"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot road
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
    
    # Plot waypoints with labels
    points_array = np.array(waypoints)
    if len(waypoints) == 4:
        labels = ['START', 'MIDDLE 1', 'MIDDLE 2', 'END']
        colors = ['green', 'blue', 'blue', 'red']
        
        for i, (x, y, label, color) in enumerate(zip(points_array[:, 0], points_array[:, 1], labels, colors)):
            ax.plot(x, y, 'o', color=color, markersize=10, markeredgecolor='black', markeredgewidth=2)
            ax.annotate(f'{i+1}: {label}', (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold', color=color)
    else:
        ax.plot(points_array[:, 0], points_array[:, 1], 'ro-', markersize=8, linewidth=2, label='Waypoints')
        
        # Add waypoint numbers
        for i, (x, y) in enumerate(waypoints):
            ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold', color='red')
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'g--', linewidth=2, alpha=0.8, label='Cubic Trajectory')
    
    ax.set_xlabel('X [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=12, fontweight='bold')
    ax.set_title(f'{title} (Speed: {speed} m/s, Horizon: {horizon})', fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save plot
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
    plt.close()
    
    return temp_file.name

def parse_waypoints(waypoints_str):
    """Parse waypoints from string"""
    points = []
    for line in waypoints_str.strip().split('\n'):
        if line.strip() and ',' in line:
            x, y = map(float, line.strip().split(','))
            points.append([x, y])
    return points

def validate_waypoints(points, required_points=None):
    """Validate waypoints"""
    if required_points and len(points) != required_points:
        return False, f"Exactly {required_points} points required. Got {len(points)} points"
    
    # Check if points are in reasonable range
    for i, (x, y) in enumerate(points):
        if not (0 <= x <= 60 and -4 <= y <= 4):
            return False, f"Point {i+1} ({x:.1f}, {y:.1f}) is outside road bounds"
    
    return True, "Valid waypoints"
