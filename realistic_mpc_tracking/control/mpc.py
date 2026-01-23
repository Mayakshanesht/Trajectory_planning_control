import numpy as np
import scipy.sparse as sp
import osqp

class MPC:
    def __init__(self, N, Q, R, Rj, umin, umax):
        self.N = N
        self.Q = np.diag(Q)
        self.R = np.diag(R)
        self.Rj = np.diag(Rj)
        self.umin = np.array(umin)
        self.umax = np.array(umax)
        self.last_u = np.zeros(len(umin))

    def solve(self, x0, ref, A, B, c, track_cons, obs_cons):
        nx, nu = len(x0), len(self.umin)
        N = self.N

        # Cost matrix for 3-second horizon
        P = sp.block_diag([
            sp.kron(sp.eye(N), self.Q),
            sp.kron(sp.eye(N), self.R),
            sp.kron(sp.eye(N-1), self.Rj)  # N-1 jerk terms for N control steps
        ])

        q = np.zeros(P.shape[0])
        for k in range(N):
            q[k*nx:(k+1)*nx] = -self.Q @ ref[k]

        rows, l, u = [], [], []

        # Dynamics constraints
        for k in range(N):
            row = np.zeros((nx, P.shape[1]))
            if k == 0:
                row[:, :nx] = np.eye(nx)
                l.append(A[k]@x0 + c[k])
                u.append(A[k]@x0 + c[k])
            else:
                row[:, (k-1)*nx:k*nx] = -A[k]
                row[:, k*nx:(k+1)*nx] = np.eye(nx)
                l.append(c[k])
                u.append(c[k])
            row[:, N*nx + k*nu:N*nx + (k+1)*nu] = -B[k]
            rows.append(row)

        # Input bounds for N control steps
        for k in range(N):
            row = np.zeros((nu, P.shape[1]))
            row[:, N*nx + k*nu:N*nx + (k+1)*nu] = np.eye(nu)
            rows.append(row)
            l.append(self.umin)
            u.append(self.umax)

        # Track constraints
        for k, cons in enumerate(track_cons):
            for n, b in cons:
                row = np.zeros(P.shape[1])
                row[k*nx] = n[0]
                row[k*nx+1] = n[1]
                rows.append(row)
                l.append(-np.inf)
                u.append(b)

        # Obstacle constraints
        obs_idx = 0
        for k in range(N):
            for obs_idx_local in range(len(obs_cons) // N):
                if obs_idx < len(obs_cons):
                    n, b = obs_cons[obs_idx]
                    row = np.zeros(P.shape[1])
                    row[k*nx] = n[0]
                    row[k*nx+1] = n[1]
                    rows.append(row)
                    l.append(b)
                    u.append(np.inf)
                    obs_idx += 1

        Aqp = sp.csr_matrix(np.vstack(rows))
        lqp = np.hstack(l)
        uqp = np.hstack(u)

        solver = osqp.OSQP()
        solver.setup(P, q, Aqp, lqp, uqp, warm_start=True, verbose=False)
        res = solver.solve()

        # Return full control sequence (N steps) for single-step application
        u = res.x[N*nx:N*nx + nu*N].reshape(N, nu)
        self.last_u = u
        return u.flatten()
