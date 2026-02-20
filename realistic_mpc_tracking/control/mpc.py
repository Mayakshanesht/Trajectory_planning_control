import numpy as np
import scipy.sparse as sp
import osqp

class MPC:
    """
    Linear time-varying MPC cast as a QP and solved by OSQP.

    Decision vector z is stacked as:
    [x_0, ..., x_{N-1}, u_0, ..., u_{N-1}, (u_1-u_0), ..., (u_{N-1}-u_{N-2})]
    """

    def __init__(self, N, Q, R, Rj, umin, umax):
        self.N = N
        self.Q = np.diag(Q)
        self.R = np.diag(R)
        self.Rj = np.diag(Rj)
        self.umin = np.array(umin)
        self.umax = np.array(umax)
        self.last_u = np.zeros(len(umin))

    def _build_cost(self, nx, nu, ref):
        """
        Build QP objective:
        sum ||x_k - x_ref,k||_Q^2 + ||u_k||_R^2 + ||u_{k+1} - u_k||_Rj^2
        """
        state_block = sp.kron(sp.eye(self.N), self.Q)
        input_block = sp.kron(sp.eye(self.N), self.R)
        jerk_block = sp.kron(sp.eye(self.N - 1), self.Rj)
        P = sp.block_diag([state_block, input_block, jerk_block], format="csc")

        q = np.zeros(P.shape[0])
        for k in range(self.N):
            state_slice = slice(k * nx, (k + 1) * nx)
            q[state_slice] = -self.Q @ ref[k]

        return P, q

    def _build_dynamics_constraints(self, x0, A, B, c, nx, nu, total_vars):
        """
        Enforce x_{k+1} = A_k x_k + B_k u_k + c_k in lifted form.
        For k=0 we anchor x_0 to linearized one-step prediction from x0.
        """
        rows, lower, upper = [], [], []
        u_offset = self.N * nx

        for k in range(self.N):
            row = np.zeros((nx, total_vars))
            if k == 0:
                row[:, :nx] = np.eye(nx)
                rhs = A[k] @ x0 + B[k] @ np.zeros(nu) + c[k]
                row[:, u_offset:u_offset + nu] = -B[k]
            else:
                prev_state = slice((k - 1) * nx, k * nx)
                curr_state = slice(k * nx, (k + 1) * nx)
                curr_input = slice(u_offset + k * nu, u_offset + (k + 1) * nu)
                row[:, prev_state] = -A[k]
                row[:, curr_state] = np.eye(nx)
                row[:, curr_input] = -B[k]
                rhs = c[k]

            rows.append(row)
            lower.append(rhs)
            upper.append(rhs)

        return rows, lower, upper

    def _build_input_constraints(self, nx, nu, total_vars):
        """Box constraints u_min <= u_k <= u_max."""
        rows, lower, upper = [], [], []
        u_offset = self.N * nx

        for k in range(self.N):
            row = np.zeros((nu, total_vars))
            input_slice = slice(u_offset + k * nu, u_offset + (k + 1) * nu)
            row[:, input_slice] = np.eye(nu)
            rows.append(row)
            lower.append(self.umin)
            upper.append(self.umax)

        return rows, lower, upper

    def _build_jerk_constraints(self, nx, nu, total_vars):
        """
        Define jerk variables as delta_u_k = u_{k+1} - u_k.
        No explicit bounds are added here; cost on delta_u regularizes changes.
        """
        rows, lower, upper = [], [], []
        u_offset = self.N * nx
        du_offset = u_offset + self.N * nu

        for k in range(self.N - 1):
            row = np.zeros((nu, total_vars))
            u_k = slice(u_offset + k * nu, u_offset + (k + 1) * nu)
            u_k1 = slice(u_offset + (k + 1) * nu, u_offset + (k + 2) * nu)
            du_k = slice(du_offset + k * nu, du_offset + (k + 1) * nu)

            row[:, u_k1] = np.eye(nu)
            row[:, u_k] = -np.eye(nu)
            row[:, du_k] = -np.eye(nu)

            rows.append(row)
            lower.append(np.zeros(nu))
            upper.append(np.zeros(nu))

        return rows, lower, upper

    def _build_track_constraints(self, track_cons, nx, total_vars):
        """
        Half-space track constraints n^T [x_k, y_k] <= b for each horizon step.
        """
        rows, lower, upper = [], [], []
        for k, cons in enumerate(track_cons):
            for n, b in cons:
                row = np.zeros(total_vars)
                row[k * nx] = n[0]
                row[k * nx + 1] = n[1]
                rows.append(row)
                lower.append(-np.inf)
                upper.append(b)
        return rows, lower, upper

    def _build_obstacle_constraints(self, obs_cons, nx, total_vars):
        """Optional obstacle half-space constraints n^T p >= b."""
        rows, lower, upper = [], [], []
        for k, cons in enumerate(obs_cons):
            for n, b in cons:
                row = np.zeros(total_vars)
                row[k * nx] = n[0]
                row[k * nx + 1] = n[1]
                rows.append(row)
                lower.append(b)
                upper.append(np.inf)
        return rows, lower, upper

    def solve(self, x0, ref, A, B, c, track_cons, obs_cons):
        nx, nu = len(x0), len(self.umin)
        total_vars = self.N * nx + self.N * nu + (self.N - 1) * nu

        if len(ref) < self.N:
            raise ValueError(f"Reference length {len(ref)} is shorter than horizon {self.N}.")

        if len(A) != self.N or len(B) != self.N or len(c) != self.N:
            raise ValueError("Linearized model lists A, B, c must each have length N.")

        P, q = self._build_cost(nx, nu, ref)

        rows, l, u = [], [], []
        builders = [
            self._build_dynamics_constraints(x0, A, B, c, nx, nu, total_vars),
            self._build_input_constraints(nx, nu, total_vars),
            self._build_jerk_constraints(nx, nu, total_vars),
            self._build_track_constraints(track_cons, nx, total_vars),
            self._build_obstacle_constraints(obs_cons, nx, total_vars),
        ]

        for block_rows, block_l, block_u in builders:
            rows.extend(block_rows)
            l.extend(block_l)
            u.extend(block_u)

        Aqp = sp.csc_matrix(np.vstack(rows))
        lqp = np.hstack(l)
        uqp = np.hstack(u)

        solver = osqp.OSQP()
        solver.setup(P=P, q=q, A=Aqp, l=lqp, u=uqp, warm_start=True, verbose=False)
        res = solver.solve()

        if res.x is None or res.info.status_val not in (1, 2):
            fallback = np.tile(self.last_u, self.N)
            return fallback

        u_sequence = res.x[self.N * nx:self.N * nx + nu * self.N].reshape(self.N, nu)
        self.last_u = u_sequence[0].copy()
        return u_sequence.flatten()
