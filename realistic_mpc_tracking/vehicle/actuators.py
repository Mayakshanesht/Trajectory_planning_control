import numpy as np

class ActuatorModel:
    def __init__(self, steer_rate, steer_max, accel_max):
        self.steer_rate = steer_rate
        self.steer_max = steer_max
        self.accel_max = accel_max
        self.u = np.zeros(2)

    def apply(self, u_cmd, dt):
        du = np.clip(
            u_cmd - self.u,
            [-self.steer_rate*dt, -self.accel_max*dt],
            [ self.steer_rate*dt,  self.accel_max*dt]
        )
        self.u += du
        self.u[0] = np.clip(self.u[0], -self.steer_max, self.steer_max)
        self.u[1] = np.clip(self.u[1], -self.accel_max, self.accel_max)
        return self.u.copy()
