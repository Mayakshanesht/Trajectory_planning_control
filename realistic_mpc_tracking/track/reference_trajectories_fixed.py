"""
Fixed reference trajectory generators with proper polynomial interpolation
"""
import numpy as np

def cubic_reference_trajectory(x, center, waypoints, N, dt, speed=2.5, current_idx=None):
    """
    Generate reference trajectory using proper cubic polynomial interpolation through user-defined waypoints
    
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
        
        # Calculate progress along the path (0 to 1)
        progress_ratio = target_idx / len(center)
        
        # Use proper cubic polynomial interpolation through waypoints
        if len(waypoints) >= 4:
            # Cubic polynomial (3rd degree) for 4 points
            # Parameterize the curve using arc length parameter t
            t_values = np.linspace(0, 1, len(waypoints))
            
            # Fit x(t) and y(t) separately as cubic polynomials
            coeffs_x = np.polyfit(t_values, waypoints[:, 0], 3)
            coeffs_y = np.polyfit(t_values, waypoints[:, 1], 3)
            
            # Evaluate polynomials at current progress
            target_x = np.polyval(coeffs_x, progress_ratio)
            target_y = np.polyval(coeffs_y, progress_ratio)
            
        elif len(waypoints) == 3:
            # Quadratic polynomial (2nd degree) for 3 points
            t_values = np.linspace(0, 1, len(waypoints))
            coeffs_x = np.polyfit(t_values, waypoints[:, 0], 2)
            coeffs_y = np.polyfit(t_values, waypoints[:, 1], 2)
            
            target_x = np.polyval(coeffs_x, progress_ratio)
            target_y = np.polyval(coeffs_y, progress_ratio)
            
        else:
            # Linear interpolation for 2 points
            t_values = np.linspace(0, 1, len(waypoints))
            coeffs_x = np.polyfit(t_values, waypoints[:, 0], 1)
            coeffs_y = np.polyfit(t_values, waypoints[:, 1], 1)
            
            target_x = np.polyval(coeffs_x, progress_ratio)
            target_y = np.polyval(coeffs_y, progress_ratio)
        
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

def test_fixed_cubic_trajectory():
    """Test the fixed cubic trajectory function"""
    print("🧪 Testing Fixed Cubic Trajectory Generation...")
    
    # Create track
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import directly from the track module
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from track.track import sinusoidal_track
    center, width = sinusoidal_track(length=60.0, amplitude=5.0, width=7.4, points=250)
    
    # Define 4 test points (START, middle1, middle2, END)
    waypoints = [
        [0.0, 0.0],      # START
        [20.0, 2.0],     # Middle 1
        [40.0, -1.0],    # Middle 2
        [60.0, 0.0]      # END
    ]
    
    print(f"📍 Test waypoints: {len(waypoints)} points")
    for i, point in enumerate(waypoints):
        print(f"  Point {i+1}: ({point[0]:.1f}, {point[1]:.1f})")
    
    try:
        # Generate trajectory
        print("\n🎯 Generating fixed cubic trajectory...")
        sample_x = np.array([waypoints[0][0], waypoints[0][1], 0, 4.0, 0, 0])
        sample_ref = cubic_reference_trajectory(
            sample_x, center, waypoints, 100, DT, 4.0, current_idx=0
        )
        
        print(f"✅ Trajectory generated: {len(sample_ref)} points")
        
        # Verify trajectory passes through waypoints
        print("\n🔍 Verifying trajectory passes through waypoints...")
        
        # Check if trajectory contains the waypoints
        trajectory_points = sample_ref[:, :2]  # X, Y coordinates
        
        for i, waypoint in enumerate(waypoints):
            # Find closest point in trajectory
            distances = np.linalg.norm(trajectory_points - waypoint, axis=1)
            closest_idx = np.argmin(distances)
            closest_point = trajectory_points[closest_idx]
            distance = distances[closest_idx]
            
            print(f"  Waypoint {i+1} ({waypoint[0]:.1f}, {waypoint[1]:.1f}):")
            print(f"    Closest trajectory point: ({closest_point[0]:.1f}, {closest_point[1]:.1f})")
            print(f"    Distance: {distance:.3f} m")
            
            if distance > 0.5:  # If more than 0.5m away, something is wrong
                print(f"    ⚠️  WARNING: Large deviation from waypoint!")
            else:
                print(f"    ✅ Good fit")
        
        # Test polynomial coefficients
        print("\n🔬 Testing polynomial coefficients...")
        t_values = np.linspace(0, 1, len(waypoints))
        coeffs_x = np.polyfit(t_values, waypoints[:, 0], 3)  # Cubic fit for x
        coeffs_y = np.polyfit(t_values, waypoints[:, 1], 3)  # Cubic fit for y
        
        print(f"  X polynomial coefficients (x = at³ + bt² + ct + d):")
        print(f"    a = {coeffs_x[0]:.6f}")
        print(f"    b = {coeffs_x[1]:.6f}")
        print(f"    c = {coeffs_x[2]:.6f}")
        print(f"    d = {coeffs_x[3]:.6f}")
        
        print(f"  Y polynomial coefficients (y = at³ + bt² + ct + d):")
        print(f"    a = {coeffs_y[0]:.6f}")
        print(f"    b = {coeffs_y[1]:.6f}")
        print(f"    c = {coeffs_y[2]:.6f}")
        print(f"    d = {coeffs_y[3]:.6f}")
        
        # Verify polynomial passes through waypoints
        print("\n🔍 Verifying polynomial passes through waypoints:")
        for i, (t, waypoint) in enumerate(zip(t_values, waypoints)):
            x_poly = np.polyval(coeffs_x, t)
            y_poly = np.polyval(coeffs_y, t)
            x_actual = waypoint[0]
            y_actual = waypoint[1]
            error_x = abs(x_poly - x_actual)
            error_y = abs(y_poly - y_actual)
            
            print(f"  Point {i+1}: t={t:.2f}")
            print(f"    X: actual={x_actual:.2f}, poly={x_poly:.2f}, error={error_x:.6f}")
            print(f"    Y: actual={y_actual:.2f}, poly={y_poly:.2f}, error={error_y:.6f}")
            
            if error_x > 0.001 or error_y > 0.001:  # More than 1mm error
                print(f"    ⚠️  WARNING: Polynomial doesn't pass through point!")
            else:
                print(f"    ✅ Perfect fit")
        
        print("\n🎉 Fixed cubic trajectory test completed!")
        return True, sample_ref
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"🐛 Traceback: {traceback.format_exc()}")
        return False, None

if __name__ == "__main__":
    success, trajectory = test_fixed_cubic_trajectory()
    
    if success:
        # Save the trajectory for debugging
        np.save('fixed_trajectory.npy', trajectory)
        print("✅ Trajectory saved as 'fixed_trajectory.npy'")
    else:
        print("❌ Test failed")
