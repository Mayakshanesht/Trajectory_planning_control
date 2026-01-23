import numpy as np

def measure(x, noise):
    return np.array([
        x[5] + np.random.randn()*noise["yaw_rate"],
        x[3] + np.random.randn()*noise["speed"],
        x[0] + np.random.randn()*noise["pos"],
        x[1] + np.random.randn()*noise["pos"]
    ])
