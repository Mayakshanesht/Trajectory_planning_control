import numpy as np
from vehicle.dynamics import step

class EKF:
    def __init__(self, dt, params):
        self.dt = dt
        self.p = params
        self.x = np.zeros(6)
        self.P = np.eye(6) * 0.2
        self.Q = np.eye(6) * 0.02
        self.R = np.eye(4) * 0.05

    def predict(self, u):
        self.x = step(self.x, u, self.dt, self.p)
        self.P += self.Q

    def update(self, z):
        H = np.zeros((4,6))
        H[0,5] = 1.0
        H[1,3] = 1.0
        H[2,0] = 1.0
        H[3,1] = 1.0

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        return self.x.copy()
