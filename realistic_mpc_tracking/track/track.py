import numpy as np

def circular_track(radius=8.0, width=2.0, points=300):
    theta = np.linspace(0, 2*np.pi, points)
    center = np.vstack([radius*np.cos(theta), radius*np.sin(theta)]).T
    return center, width

def sinusoidal_track(length=50.0, amplitude=4.0, width=3.0, points=200):
    """
    Create a sinusoidal two-lane road
    
    Args:
        length: Total length of the road (meters)
        amplitude: Amplitude of the sinusoid (meters)
        width: Width of the road (meters)
        points: Number of points along the centerline
    
    Returns:
        center: Road centerline points [N, 2]
        width: Road width
    """
    x = np.linspace(0, length, points)
    y = amplitude * np.sin(2 * np.pi * x / length)
    center = np.vstack([x, y]).T
    return center, width

def corridor_constraints(center, width, k):
    p0 = center[k]
    p1 = center[(k+1) % len(center)]
    t = p1 - p0
    t = t / np.linalg.norm(t)
    n = np.array([-t[1], t[0]])

    left_b = n @ p0 + width/2
    right_b = -n @ p0 + width/2

    return [(n, left_b), (-n, right_b)]
