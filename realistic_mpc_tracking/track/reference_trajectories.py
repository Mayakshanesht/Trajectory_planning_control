"""
Reference trajectory generators for different path patterns
"""
import numpy as np

def sinusoidal_reference_trajectory_with_lane_changes(x, center, N, dt, speed=2.5, target_lane_offset=0.0, lane_width=3.7, current_idx=None):
    """
    Generate reference trajectory with lane changes
    
    Args:
        x: Current state [X, Y, psi, vx, vy, r]
        center: Road centerline points
        N: Number of prediction steps
        dt: Time step
        speed: Reference speed
        target_lane_offset: Target offset from centerline (positive = right lane)
        lane_width: Width of each lane
        current_idx: Current progress index (optional)
    
    Returns:
        ref: Reference trajectory [N, 6]
    """
    ref = []
    
    # Find current position along road if not provided
    if current_idx is None:
        current_pos = x[:2]
        distances = np.linalg.norm(center - current_pos, axis=1)
        current_idx = np.argmin(distances)
    
    # Enhanced lookahead calculation for better progress
    base_lookahead = max(3, int(speed * dt * 5))
    
    for k in range(N):
        # Adaptive lookahead based on position in trajectory
        adaptive_lookahead = base_lookahead + k // 2
        
        # Always move forward, never backward
        target_idx = current_idx + adaptive_lookahead
        
        # Handle end of road - stop at end, don't wrap
        if target_idx >= len(center):
            target_idx = len(center) - 1
            target_speed = max(0.1, speed * 0.1)  # Very slow at end
        else:
            target_speed = speed
        
        # Get centerline position
        center_point = center[target_idx]
        
        # Calculate road normal at this point
        if target_idx < len(center) - 1:
            tangent = center[target_idx + 1] - center[target_idx]
        else:
            tangent = center[target_idx] - center[target_idx - 1]
        
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        
        # Apply lane offset with smooth transition
        current_offset = target_lane_offset
        if k == 0:  # First point - blend from current position
            if current_idx < len(center) - 1:
                current_normal = np.array([-center[current_idx + 1, 1] + center[current_idx, 1], 
                                           center[current_idx + 1, 0] - center[current_idx, 0]])
                current_normal = current_normal / np.linalg.norm(current_normal)
                current_pos_offset = (x[:2] - center[current_idx]) @ current_normal
                # Smooth blend to target lane
                blend_factor = min(1.0, k * 0.2)  # Faster blending
                current_offset = current_pos_offset * (1 - blend_factor) + target_lane_offset * blend_factor
        elif k < 5:  # Early trajectory points - smooth transition
            blend_factor = k / 5.0
            current_offset = current_offset * blend_factor + target_lane_offset * (1 - blend_factor)
        
        X, Y = center_point + normal * current_offset
        
        # Calculate proper heading based on road direction
        if target_idx < len(center) - 1:
            dx = center[target_idx+1, 0] - center[target_idx, 0]
            dy = center[target_idx+1, 1] - center[target_idx, 1]
        else:
            # At the end, use the last direction
            dx = center[target_idx, 0] - center[target_idx-1, 0]
            dy = center[target_idx, 1] - center[target_idx-1, 1]
        
        psi = np.arctan2(dy, dx)
        ref.append([X, Y, psi, target_speed, 0.0, 0.0])

    return np.array(ref)

def zigzag_reference_trajectory(x, center, N, dt, speed=2.5, amplitude_offset=2.0, current_idx=None):
    """
    Generate zigzag reference trajectory on the same sinusoidal road
    
    Args:
        x: Current state [X, Y, psi, vx, vy, r]
        center: Road centerline points
        N: Number of prediction steps
        dt: Time step
        speed: Reference speed
        amplitude_offset: How far to deviate from centerline (meters)
        current_idx: Current progress index (optional)
    
    Returns:
        ref: Reference trajectory [N, 6]
    """
    ref = []
    
    # Find current position along road if not provided
    if current_idx is None:
        current_pos = x[:2]
        distances = np.linalg.norm(center - current_pos, axis=1)
        current_idx = np.argmin(distances)
    
    # Enhanced lookahead calculation for better progress
    base_lookahead = max(3, int(speed * dt * 5))
    
    for k in range(N):
        # Adaptive lookahead based on position in trajectory
        adaptive_lookahead = base_lookahead + k // 2
        
        # Always move forward, never backward
        target_idx = current_idx + adaptive_lookahead
        
        # Handle end of road - stop at end, don't wrap
        if target_idx >= len(center):
            target_idx = len(center) - 1
            target_speed = max(0.1, speed * 0.1)  # Very slow at end
        else:
            target_speed = speed
        
        # Get centerline position
        center_point = center[target_idx]
        
        # Calculate road normal at this point
        if target_idx < len(center) - 1:
            tangent = center[target_idx + 1] - center[target_idx]
        else:
            tangent = center[target_idx] - center[target_idx - 1]
        
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        
        # Create zigzag pattern by alternating sides
        cycle_position = (target_idx // 10) % 4  # Change every 10 points, 4-cycle pattern
        if cycle_position == 0:
            offset = amplitude_offset  # Right side
        elif cycle_position == 1:
            offset = 0  # Center
        elif cycle_position == 2:
            offset = -amplitude_offset  # Left side
        else:
            offset = 0  # Center
        
        # Smooth transition to target offset
        current_offset = offset  # Initialize with target offset
        if k == 0:  # First point - blend from current position
            if current_idx < len(center) - 1:
                current_normal = np.array([-center[current_idx + 1, 1] + center[current_idx, 1], 
                                           center[current_idx + 1, 0] - center[current_idx, 0]])
                current_normal = current_normal / np.linalg.norm(current_normal)
                current_pos_offset = (x[:2] - center[current_idx]) @ current_normal
                # Smooth blend to target offset
                blend_factor = min(1.0, k * 0.2)  # Faster blending
                current_offset = current_pos_offset * (1 - blend_factor) + offset * blend_factor
        elif k < 5:  # Early trajectory points - smooth transition
            blend_factor = k / 5.0
            current_offset = current_offset * blend_factor + offset * (1 - blend_factor)
        else:
            current_offset = offset
        
        X, Y = center_point + normal * current_offset
        
        # Calculate proper heading based on road direction
        if target_idx < len(center) - 1:
            dx = center[target_idx+1, 0] - center[target_idx, 0]
            dy = center[target_idx+1, 1] - center[target_idx, 1]
        else:
            # At the end, use the last direction
            dx = center[target_idx, 0] - center[target_idx-1, 0]
            dy = center[target_idx, 1] - center[target_idx-1, 1]
        
        psi = np.arctan2(dy, dx)
        ref.append([X, Y, psi, target_speed, 0.0, 0.0])

    return np.array(ref)

def s_curve_reference_trajectory(x, center, N, dt, speed=2.5, current_idx=None):
    """
    Generate S-curve reference trajectory with speed variations on the same sinusoidal road
    
    Args:
        x: Current state [X, Y, psi, vx, vy, r]
        center: Road centerline points
        N: Number of prediction steps
        dt: Time step
        speed: Reference speed
        current_idx: Current progress index (optional)
    
    Returns:
        ref: Reference trajectory [N, 6]
    """
    ref = []
    
    # Find current position along road if not provided
    if current_idx is None:
        current_pos = x[:2]
        distances = np.linalg.norm(center - current_pos, axis=1)
        current_idx = np.argmin(distances)
    
    # Enhanced lookahead calculation for better progress
    base_lookahead = max(3, int(speed * dt * 5))
    
    for k in range(N):
        # Adaptive lookahead based on position in trajectory
        adaptive_lookahead = base_lookahead + k // 2
        
        # Always move forward, never backward
        target_idx = current_idx + adaptive_lookahead
        
        # Handle end of road - stop at end, don't wrap
        if target_idx >= len(center):
            target_idx = len(center) - 1
            target_speed = max(0.1, speed * 0.1)  # Very slow at end
        else:
            target_speed = speed
        
        # Get centerline position
        center_point = center[target_idx]
        
        # Calculate road normal at this point
        if target_idx < len(center) - 1:
            tangent = center[target_idx + 1] - center[target_idx]
        else:
            tangent = center[target_idx] - center[target_idx - 1]
        
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        
        # Create S-curve pattern with smooth transitions
        progress_ratio = target_idx / len(center)
        
        # S-curve: smooth transition from left to center to right and back
        if progress_ratio < 0.25:
            # First quarter: gradual move from left to center
            blend = progress_ratio / 0.25
            offset = -1.5 * (1 - blend)  # Start at -1.5m (left), move to 0 (center)
        elif progress_ratio < 0.5:
            # Second quarter: gradual move from center to right
            blend = (progress_ratio - 0.25) / 0.25
            offset = 1.5 * blend  # Move from 0 (center) to +1.5m (right)
        elif progress_ratio < 0.75:
            # Third quarter: gradual move from right back to center
            blend = (progress_ratio - 0.5) / 0.25
            offset = 1.5 * (1 - blend)  # Move from +1.5m (right) to 0 (center)
        else:
            # Final quarter: gradual move from center to left
            blend = (progress_ratio - 0.75) / 0.25
            offset = -1.5 * blend  # Move from 0 (center) to -1.5m (left)
        
        # Add small sinusoidal variation for more realistic path
        sinusoidal_variation = 0.3 * np.sin(4 * np.pi * progress_ratio)
        offset += sinusoidal_variation
        
        # Smooth transition to target offset
        current_offset = offset  # Initialize with target offset
        if k == 0:  # First point - blend from current position
            if current_idx < len(center) - 1:
                current_normal = np.array([-center[current_idx + 1, 1] + center[current_idx, 1], 
                                           center[current_idx + 1, 0] - center[current_idx, 0]])
                current_normal = current_normal / np.linalg.norm(current_normal)
                current_pos_offset = (x[:2] - center[current_idx]) @ current_normal
                # Smooth blend to target offset
                blend_factor = min(1.0, k * 0.2)  # Faster blending
                current_offset = current_pos_offset * (1 - blend_factor) + offset * blend_factor
        elif k < 5:  # Early trajectory points - smooth transition
            blend_factor = k / 5.0
            current_offset = current_offset * blend_factor + offset * (1 - blend_factor)
        else:
            current_offset = offset
        
        X, Y = center_point + normal * current_offset
        
        # Calculate proper heading based on road direction
        if target_idx < len(center) - 1:
            dx = center[target_idx+1, 0] - center[target_idx, 0]
            dy = center[target_idx+1, 1] - center[target_idx, 1]
        else:
            # At the end, use the last direction
            dx = center[target_idx, 0] - center[target_idx-1, 0]
            dy = center[target_idx, 1] - center[target_idx-1, 1]
        
        psi = np.arctan2(dy, dx)
        ref.append([X, Y, psi, target_speed, 0.0, 0.0])

    return np.array(ref)

def polynomial_reference_trajectory_quarticic(x, center, waypoints, N, dt, speed=2.5, current_idx=None):
    """
    Generate reference trajectory using 4th degree polynomial interpolation through user-defined waypoints
    
    Args:
        x: Current state [X, Y, psi, vx, vy, r]
        center: Road centerline points
        waypoints: List of (x, y) waypoints for the reference path
        N: Number of prediction steps
        dt: Time step
        speed: Reference speed
        current_idx: Current progress index (optional)
    
    Returns:
        ref: Reference trajectory [N, 6]
    """
    ref = []
    
    if len(waypoints) < 2:
        raise ValueError("At least 2 waypoints required for polynomial interpolation")
    
    # Convert waypoints to numpy array
    waypoints = np.array(waypoints)
    
    # Find current position along road if not provided
    if current_idx is None:
        current_pos = x[:2]
        distances = np.linalg.norm(center - current_pos, axis=1)
        current_idx = np.argmin(distances)
    
    # Enhanced lookahead calculation for better progress
    base_lookahead = max(3, int(speed * dt * 5))
    
    for k in range(N):
        # Adaptive lookahead based on position in trajectory
        adaptive_lookahead = base_lookahead + k // 2
        
        # Always move forward, never backward
        target_idx = current_idx + adaptive_lookahead
        
        # Handle end of road - stop at end, don't wrap
        if target_idx >= len(center):
            target_idx = len(center) - 1
            target_speed = max(0.1, speed * 0.1)  # Very slow at end
        else:
            target_speed = speed
        
        # Get centerline position
        center_point = center[target_idx]
        
        # Calculate road normal at this point
        if target_idx < len(center) - 1:
            tangent = center[target_idx + 1] - center[target_idx]
        else:
            tangent = center[target_idx] - center[target_idx - 1]
        
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        
        # Calculate progress along the path
        progress_ratio = target_idx / len(center)
        
        # Use 4th degree polynomial interpolation to get offset from centerline
        if len(waypoints) >= 5:
            # Quartic polynomial for very smooth curves
            t_values = np.linspace(0, 1, len(waypoints))
            coeffs = np.polyfit(t_values, waypoints[:, 1], 4)  # Fit y as function of x
            target_x = waypoints[0, 0] + progress_ratio * (waypoints[-1, 0] - waypoints[0, 0])
            target_y = np.polyval(coeffs, target_x)
        elif len(waypoints) >= 4:
            # Cubic polynomial for smooth curves
            t_values = np.linspace(0, 1, len(waypoints))
            coeffs = np.polyfit(t_values, waypoints[:, 1], 3)  # Fit y as function of x
            target_x = waypoints[0, 0] + progress_ratio * (waypoints[-1, 0] - waypoints[0, 0])
            target_y = np.polyval(coeffs, target_x)
        else:
            # Linear interpolation for 2 points
            t_values = np.linspace(0, 1, len(waypoints))
            coeffs = np.polyfit(t_values, waypoints[:, 1], 1)  # Linear fit
            target_x = waypoints[0, 0] + progress_ratio * (waypoints[-1, 0] - waypoints[0, 0])
            target_y = np.polyval(coeffs, target_x)
        
        # Calculate offset from centerline
        target_point = np.array([target_x, target_y])
        offset_vector = target_point - center_point
        offset = np.dot(offset_vector, normal)
        
        # Smooth transition to target offset
        current_offset = offset
        if k == 0:  # First point - blend from current position
            if current_idx < len(center) - 1:
                current_normal = np.array([-center[current_idx + 1, 1] + center[current_idx, 1], 
                                           center[current_idx + 1, 0] - center[current_idx, 0]])
                current_normal = current_normal / np.linalg.norm(current_normal)
                current_pos_offset = (x[:2] - center[current_idx]) @ current_normal
                # Smooth blend to target offset
                blend_factor = min(1.0, k * 0.2)  # Faster blending
                current_offset = current_pos_offset * (1 - blend_factor) + offset * blend_factor
        elif k < 5:  # Early trajectory points - smooth transition
            blend_factor = k / 5.0
            current_offset = current_offset * blend_factor + offset * (1 - blend_factor)
        else:
            current_offset = offset
        
        X, Y = center_point + normal * current_offset
        
        # Calculate proper heading based on road direction
        if target_idx < len(center) - 1:
            dx = center[target_idx+1, 0] - center[target_idx, 0]
            dy = center[target_idx+1, 1] - center[target_idx, 1]
        else:
            # At the end, use the last direction
            dx = center[target_idx, 0] - center[target_idx-1, 0]
            dy = center[target_idx, 1] - center[target_idx-1, 1]
        
        psi = np.arctan2(dy, dx)
        ref.append([X, Y, psi, target_speed, 0.0, 0.0])

    return np.array(ref)

def polynomial_reference_trajectory(x, center, waypoints, N, dt, speed=2.5, current_idx=None):
    """
    Generate reference trajectory using polynomial interpolation through user-defined waypoints
    
    Args:
        x: Current state [X, Y, psi, vx, vy, r]
        center: Road centerline points
        waypoints: List of (x, y) waypoints for the reference path
        N: Number of prediction steps
        dt: Time step
        speed: Reference speed
        current_idx: Current progress index (optional)
    
    Returns:
        ref: Reference trajectory [N, 6]
    """
    ref = []
    
    if len(waypoints) < 2:
        raise ValueError("At least 2 waypoints required for polynomial interpolation")
    
    # Convert waypoints to numpy array
    waypoints = np.array(waypoints)
    
    # Find current position along road if not provided
    if current_idx is None:
        current_pos = x[:2]
        distances = np.linalg.norm(center - current_pos, axis=1)
        current_idx = np.argmin(distances)
    
    # Enhanced lookahead calculation for better progress
    base_lookahead = max(3, int(speed * dt * 5))
    
    for k in range(N):
        # Adaptive lookahead based on position in trajectory
        adaptive_lookahead = base_lookahead + k // 2
        
        # Always move forward, never backward
        target_idx = current_idx + adaptive_lookahead
        
        # Handle end of road - stop at end, don't wrap
        if target_idx >= len(center):
            target_idx = len(center) - 1
            target_speed = max(0.1, speed * 0.1)  # Very slow at end
        else:
            target_speed = speed
        
        # Get centerline position
        center_point = center[target_idx]
        
        # Calculate road normal at this point
        if target_idx < len(center) - 1:
            tangent = center[target_idx + 1] - center[target_idx]
        else:
            tangent = center[target_idx] - center[target_idx - 1]
        
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        
        # Calculate progress along the path
        progress_ratio = target_idx / len(center)
        
        # Use polynomial interpolation to get offset from centerline
        if len(waypoints) >= 3:
            # Cubic polynomial for smooth curves
            t_values = np.linspace(0, 1, len(waypoints))
            coeffs = np.polyfit(t_values, waypoints[:, 1], 3)  # Fit y as function of x
            target_x = waypoints[0, 0] + progress_ratio * (waypoints[-1, 0] - waypoints[0, 0])
            target_y = np.polyval(coeffs, target_x)
        else:
            # Linear interpolation for 2 points
            t_values = np.linspace(0, 1, len(waypoints))
            coeffs = np.polyfit(t_values, waypoints[:, 1], 1)  # Linear fit
            target_x = waypoints[0, 0] + progress_ratio * (waypoints[-1, 0] - waypoints[0, 0])
            target_y = np.polyval(coeffs, target_x)
        
        # Calculate offset from centerline
        target_point = np.array([target_x, target_y])
        offset_vector = target_point - center_point
        offset = np.dot(offset_vector, normal)
        
        # Smooth transition to target offset
        current_offset = offset
        if k == 0:  # First point - blend from current position
            if current_idx < len(center) - 1:
                current_normal = np.array([-center[current_idx + 1, 1] + center[current_idx, 1], 
                                           center[current_idx + 1, 0] - center[current_idx, 0]])
                current_normal = current_normal / np.linalg.norm(current_normal)
                current_pos_offset = (x[:2] - center[current_idx]) @ current_normal
                # Smooth blend to target offset
                blend_factor = min(1.0, k * 0.2)  # Faster blending
                current_offset = current_pos_offset * (1 - blend_factor) + offset * blend_factor
        elif k < 5:  # Early trajectory points - smooth transition
            blend_factor = k / 5.0
            current_offset = current_offset * blend_factor + offset * (1 - blend_factor)
        else:
            current_offset = offset
        
        X, Y = center_point + normal * current_offset
        
        # Calculate proper heading based on road direction
        if target_idx < len(center) - 1:
            dx = center[target_idx+1, 0] - center[target_idx, 0]
            dy = center[target_idx+1, 1] - center[target_idx, 1]
        else:
            # At the end, use the last direction
            dx = center[target_idx, 0] - center[target_idx-1, 0]
            dy = center[target_idx, 1] - center[target_idx-1, 1]
        
        psi = np.arctan2(dy, dx)
        ref.append([X, Y, psi, target_speed, 0.0, 0.0])

    return np.array(ref)
