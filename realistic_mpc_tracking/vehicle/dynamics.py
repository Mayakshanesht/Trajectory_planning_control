import numpy as np

def step(x, u, dt, vehicle_params):
    """
    Simple kinematic bicycle model
    
    Args:
        x: state [X, Y, psi, vx, vy, r]
        u: control [delta, ax]
        dt: time step
        vehicle_params: vehicle parameters dictionary
    
    Returns:
        x_next: next state
    """
    X, Y, psi, vx, vy, r = x
    delta, ax = u
    
    # Vehicle parameters
    lf = vehicle_params["lf"]
    lr = vehicle_params["lr"]
    
    # Kinematic bicycle model
    beta = np.arctan(lr / (lf + lr) * np.tan(delta))
    
    dX = vx * np.cos(psi + beta)
    dY = vx * np.sin(psi + beta)
    dpsi = vx / lr * np.sin(beta)
    dvx = ax
    dvy = 0.0  # Simplified kinematic model
    dr = 0.0   # Simplified kinematic model
    
    x_next = x + dt * np.array([dX, dY, dpsi, dvx, dvy, dr])
    x_next[2] = np.arctan2(np.sin(x_next[2]), np.cos(x_next[2]))  # Normalize heading
    
    return x_next