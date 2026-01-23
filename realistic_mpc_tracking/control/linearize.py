import numpy as np
from vehicle.dynamics import step

def linearize(x, u, dt, p):
    eps = 1e-5
    nx, nu = len(x), len(u)
    f0 = step(x, u, dt, p)

    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))

    for i in range(nx):
        dx = np.zeros(nx)
        dx[i] = eps
        A[:, i] = (step(x + dx, u, dt, p) - f0) / eps

    for i in range(nu):
        du = np.zeros(nu)
        du[i] = eps
        B[:, i] = (step(x, u + du, dt, p) - f0) / eps

    c = f0 - A @ x - B @ u
    return A, B, c
