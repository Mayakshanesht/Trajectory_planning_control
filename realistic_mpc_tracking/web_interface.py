"""
Student-friendly Gradio UI for MPC trajectory tracking.

Main workflow:
1. Click exactly 4 points on the road image.
2. Fit cubic polynomials x(t), y(t) through those points.
3. Build reference states [x, y, psi, v, 0, 0].
4. Run linearized MPC with road corridor constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import gradio_client.utils as gr_client_utils
import os
import csv
import tempfile

from config.params import DT, VEHICLE, LIMITS, NOISE
from control.linearize import linearize
from control.mpc import MPC
from estimation.ekf import EKF
from track.track import sinusoidal_track, corridor_constraints
from utils.plotting import create_trajectory_gif
from vehicle.actuators import ActuatorModel
from vehicle.dynamics import step
from vehicle.sensors import measure


TRACK_LENGTH = 60.0
TRACK_AMPLITUDE = 5.0
TRACK_WIDTH = 7.4
TRACK_POINTS = 250

X_MIN, X_MAX = 0.0, 60.0
Y_MIN, Y_MAX = -8.0, 8.0
IMG_W, IMG_H = 1200, 420

CENTER, WIDTH = sinusoidal_track(
    length=TRACK_LENGTH, amplitude=TRACK_AMPLITUDE, width=TRACK_WIDTH, points=TRACK_POINTS
)

HALF_WIDTH_MARGIN = WIDTH / 2.0 - 0.35

MPC_EXPLANATION = """
### MPC Optimization Used Here
- Objective:
  - State tracking: `sum_k (x_k - x_ref,k)^T Q (x_k - x_ref,k)`
  - Control effort: `sum_k u_k^T R u_k`
  - Input smoothness: `sum_k (u_{k+1} - u_k)^T Rj (u_{k+1} - u_k)`
- System model constraint (linearized every step):
  - `x_{k+1} = A_k x_k + B_k u_k + c_k`
- Input constraints:
  - `u_min <= u_k <= u_max`
- Road corridor constraints:
  - Half-space bounds on `(x_k, y_k)` from the local road normal
- Solver:
  - All terms are assembled into one QP and solved with OSQP in `control/mpc.py`.
"""


def _patch_gradio_schema_bool_bug():
    """Work around gradio_client schema parsing on boolean additionalProperties."""
    original = gr_client_utils._json_schema_to_python_type

    def patched(schema, defs=None):
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return original(schema, defs)

    gr_client_utils._json_schema_to_python_type = patched


_patch_gradio_schema_bool_bug()


def _road_boundaries(center: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    left, right = [], []
    for i in range(len(center)):
        tangent = center[min(i + 1, len(center) - 1)] - center[max(i - 1, 0)]
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        left.append(center[i] + normal * width / 2.0)
        right.append(center[i] - normal * width / 2.0)
    return np.asarray(left), np.asarray(right)


def _center_normals(center: np.ndarray) -> np.ndarray:
    normals = []
    for i in range(len(center)):
        tangent = center[min(i + 1, len(center) - 1)] - center[max(i - 1, 0)]
        tangent = tangent / max(np.linalg.norm(tangent), 1e-8)
        normals.append(np.array([-tangent[1], tangent[0]]))
    return np.asarray(normals)


NORMALS = _center_normals(CENTER)


def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / max(b - a, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _forward_nearest_index(path_xy: np.ndarray, p_xy: np.ndarray, last_idx: int, window: int = 40) -> int:
    lo = max(0, last_idx)
    hi = min(len(path_xy), lo + window)
    local = path_xy[lo:hi]
    if len(local) == 0:
        return min(last_idx, len(path_xy) - 1)
    return int(lo + np.argmin(np.linalg.norm(local - p_xy, axis=1)))


def _resample_polyline(points: np.ndarray, samples: int = 180) -> np.ndarray:
    if len(points) < 2:
        raise ValueError("Need at least 2 points to build a trajectory.")
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.hstack(([0.0], np.cumsum(seg)))
    total = s[-1]
    if total < 1e-6:
        raise ValueError("Reference points are degenerate (zero-length path).")
    s_target = np.linspace(0.0, total, samples)
    x = np.interp(s_target, s, points[:, 0])
    y = np.interp(s_target, s, points[:, 1])
    return np.column_stack([x, y])


def _reference_from_xy(path_xy: np.ndarray, base_speed: float) -> np.ndarray:
    path_xy = _resample_polyline(path_xy, samples=200)
    dx = np.gradient(path_xy[:, 0])
    dy = np.gradient(path_xy[:, 1])
    psi = np.unwrap(np.arctan2(dy, dx))
    ds = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-5)
    kappa = np.abs(np.gradient(psi) / ds)
    speed_scale = np.clip(1.0 / (1.0 + 6.0 * kappa), 0.45, 1.0)
    v_ref = np.clip(base_speed * speed_scale, 1.0, base_speed)

    ref = np.zeros((len(path_xy), 6))
    ref[:, 0] = path_xy[:, 0]
    ref[:, 1] = path_xy[:, 1]
    ref[:, 2] = psi
    ref[:, 3] = v_ref
    return ref


def _build_offset_path(offsets: np.ndarray) -> np.ndarray:
    clamped = np.clip(offsets, -HALF_WIDTH_MARGIN, HALF_WIDTH_MARGIN)
    return CENTER + NORMALS * clamped[:, None]


def _trajectory_offsets(traj_type: str) -> np.ndarray:
    s = np.linspace(0.0, 1.0, len(CENTER))
    if traj_type == "ref1":
        left = -WIDTH * 0.22
        right = WIDTH * 0.22
        lane = left + (right - left) * _smoothstep(0.20, 0.42, s)
        lane = lane + (left - right) * _smoothstep(0.58, 0.82, s)
        return lane
    if traj_type == "ref2":
        return (WIDTH * 0.24) * np.sin(3.2 * np.pi * s)
    if traj_type == "ref3":
        return (WIDTH * 0.2) * np.sin(2.0 * np.pi * s) * (0.8 + 0.2 * np.cos(2.0 * np.pi * s))
    raise ValueError(f"Unknown trajectory type: {traj_type}")


def render_track_image(waypoints: list[list[float]] | None) -> np.ndarray:
    waypoints = waypoints or []
    left, right = _road_boundaries(CENTER, WIDTH)

    fig = plt.figure(figsize=(IMG_W / 100, IMG_H / 100), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    ax.fill_between(left[:, 0], left[:, 1], right[:, 1], color="#d6d6d6", alpha=0.65)
    ax.plot(CENTER[:, 0], CENTER[:, 1], "--", color="#5f5f5f", linewidth=1.0)
    ax.plot(left[:, 0], left[:, 1], color="black", linewidth=1.7)
    ax.plot(right[:, 0], right[:, 1], color="black", linewidth=1.7)

    labels = ["START", "MID-1", "MID-2", "END"]
    for idx, (x, y) in enumerate(waypoints):
        ax.scatter(x, y, c="#d62728", s=120, zorder=5, edgecolors="white", linewidths=1.2)
        ax.text(x + 0.35, y + 0.25, labels[idx], fontsize=9, weight="bold", color="#222222")

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return frame


def _is_on_road(x: float, y: float) -> bool:
    nearest = int(np.argmin(np.linalg.norm(CENTER - np.array([x, y]), axis=1)))
    dist = float(np.linalg.norm(CENTER[nearest] - np.array([x, y])))
    return dist <= (WIDTH / 2.0)


def _build_mpc(horizon: int) -> MPC:
    q_weights = [120, 120, 30, 15, 10, 10]
    r_weights = [12.0, 5.0]
    rj_weights = [60.0, 20.0]
    return MPC(
        horizon,
        q_weights,
        r_weights,
        rj_weights,
        [-LIMITS["steer_max"], -LIMITS["accel_max"]],
        [LIMITS["steer_max"], LIMITS["accel_max"]],
    )


def _run_mpc_on_reference(ref_traj: np.ndarray, horizon: int):
    x = np.array([ref_traj[0, 0], ref_traj[0, 1], ref_traj[0, 2], ref_traj[0, 3], 0.0, 0.0])
    u_prev = np.zeros(2)

    actuator = ActuatorModel(LIMITS["steer_rate"], LIMITS["steer_max"], LIMITS["accel_max"])
    ekf = EKF(DT, VEHICLE)
    mpc = _build_mpc(horizon)

    states, errors = [x.copy()], []
    last_ref_idx = 0
    stall_steps = 0

    max_steps = int(len(ref_traj) * 2.4)
    for _ in range(max_steps):
        idx = _forward_nearest_index(ref_traj[:, :2], x[:2], last_ref_idx, window=45)
        if idx <= last_ref_idx:
            stall_steps += 1
        else:
            stall_steps = 0
        idx = max(idx, last_ref_idx)
        last_ref_idx = idx

        if idx >= len(ref_traj) - horizon - 1:
            break
        if stall_steps > 45:
            break

        ref = ref_traj[idx : idx + horizon].copy()
        z = measure(x, NOISE)
        ekf.predict(u_prev)
        xhat = ekf.update(z)

        A_list, B_list, c_list, track_cons = [], [], [], []
        for k in range(horizon):
            A_k, B_k, c_k = linearize(ref[k], u_prev, DT, VEHICLE)
            A_list.append(A_k)
            B_list.append(B_k)
            c_list.append(c_k)
            center_idx = int(np.argmin(np.linalg.norm(CENTER - ref[k, :2], axis=1)))
            center_idx = min(center_idx, len(CENTER) - 2)
            track_cons.append(corridor_constraints(CENTER, WIDTH, center_idx))

        u_sequence = mpc.solve(xhat, ref, A_list, B_list, c_list, track_cons, [])
        u = actuator.apply(u_sequence[:2], DT)
        x = step(x, u, DT, VEHICLE)
        u_prev = u

        states.append(x.copy())
        errors.append(np.linalg.norm(x[:2] - ref[0, :2]))

    states = np.asarray(states, dtype=float)
    if len(states) < 2:
        return None, None

    ref_indices = np.linspace(0, len(ref_traj) - 1, len(states)).astype(int)
    ref_interp = ref_traj[ref_indices]

    final_target = ref_traj[last_ref_idx, :2]
    return {
        "final_error": float(np.linalg.norm(states[-1, :2] - final_target)),
        "avg_error": float(np.mean(errors)) if errors else 0.0,
        "max_error": float(np.max(errors)) if errors else 0.0,
        "path_length": float(np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1))),
        "steps": int(len(states)),
        "progress": float(last_ref_idx / max(1, len(ref_traj) - 1)),
        "states": states,
        "ref_interp": ref_interp,
    }


def _format_metrics(title: str, metrics: dict, extra_lines: list[str] | None = None) -> str:
    lines = [
        f"{title}",
        "",
        f"- Final error: {metrics['final_error']:.2f} m",
        f"- Average error: {metrics['avg_error']:.2f} m",
        f"- Max error: {metrics['max_error']:.2f} m",
        f"- Path length: {metrics['path_length']:.2f} m",
        f"- Simulation steps: {metrics['steps']}",
        f"- Completed progress: {metrics['progress'] * 100:.1f}%",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    return "\n".join(lines)


def run_reference_trajectory(traj_type, speed, horizon):
    offset_profile = _trajectory_offsets(traj_type)
    path = _build_offset_path(offset_profile)
    ref_traj = _reference_from_xy(path, speed)
    metrics = _run_mpc_on_reference(ref_traj, horizon)
    if metrics is None:
        return "MPC did not produce enough simulation steps. Try a larger horizon.", None

    gif_path = f"mpc_{traj_type}_tracking.gif"
    create_trajectory_gif(metrics["states"], metrics["ref_interp"], CENTER, WIDTH, [], save_path=gif_path, fps=20)

    report = _format_metrics(
        f"Reference Trajectory `{traj_type}` Results",
        metrics,
        [
            f"- Horizon: {horizon}",
            f"- Peak reference speed: {speed:.2f} m/s",
            "- Reference uses smooth lateral offset profile + curvature-aware speed scaling.",
        ],
    )
    return report, gif_path


def _guidance_text() -> str:
    return (
        "Reference input guide:\n"
        "1) Use either columns x,y (meters) OR s,offset.\n"
        "2) s is normalized path position in [0, 1].\n"
        f"3) offset must stay within +/-{HALF_WIDTH_MARGIN:.2f} m.\n"
        "4) Provide at least 4 points; points should move forward along the road."
    )


def _create_reference_template() -> str:
    fd, path = tempfile.mkstemp(prefix="trajectory_template_", suffix=".csv")
    os.close(fd)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["s", "offset"])
        writer.writerow([0.00, 0.00])
        writer.writerow([0.25, 1.20])
        writer.writerow([0.50, -1.20])
        writer.writerow([0.75, 1.00])
        writer.writerow([1.00, 0.00])
    return path


def _center_point_from_s_offset(s: float, offset: float) -> np.ndarray:
    s = float(np.clip(s, 0.0, 1.0))
    idx = min(int(round(s * (len(CENTER) - 1))), len(CENTER) - 1)
    offset = float(np.clip(offset, -HALF_WIDTH_MARGIN, HALF_WIDTH_MARGIN))
    return CENTER[idx] + NORMALS[idx] * offset


def _load_points_from_file(file_obj):
    if file_obj is None:
        raise ValueError("Upload a CSV/XLSX file first.")
    path = getattr(file_obj, "name", str(file_obj))
    ext = os.path.splitext(path)[1].lower()

    rows = []
    if ext == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    elif ext in (".xlsx", ".xls"):
        try:
            import pandas as pd  # Lazy import: only needed for Excel files
        except Exception as exc:
            raise ValueError(f"Excel parsing requires pandas/openpyxl in this env. ({exc})") from exc
        df = pd.read_excel(path)
        cols = {str(c).strip().lower(): c for c in df.columns}
        rows = []
        for _, r in df.iterrows():
            rows.append({k: r[v] for k, v in cols.items()})
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx")

    if not rows:
        raise ValueError("No rows found in uploaded file.")

    keys = set(rows[0].keys())
    points = []
    if {"x", "y"}.issubset(keys):
        for row in rows:
            x = float(row["x"])
            y = float(row["y"])
            points.append([x, y])
    elif {"s", "offset"}.issubset(keys):
        for row in rows:
            s = float(row["s"])
            offset = float(row["offset"])
            points.append(_center_point_from_s_offset(s, offset).tolist())
    else:
        raise ValueError("Expected columns x,y or s,offset.")

    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        raise ValueError("At least 4 points are required.")

    idxs = [int(np.argmin(np.linalg.norm(CENTER - p, axis=1))) for p in points]
    if any(b < a for a, b in zip(idxs, idxs[1:])):
        raise ValueError("Points must move forward along the road (monotonic progress).")

    for p in points:
        if not _is_on_road(float(p[0]), float(p[1])):
            raise ValueError("One or more points are outside road bounds.")
    return points


def run_uploaded_reference_mpc(file_obj, speed, horizon):
    try:
        points = _load_points_from_file(file_obj)
    except Exception as exc:
        return f"Upload validation failed: {exc}\n\n{_guidance_text()}", None

    ref_traj = _reference_from_xy(points, speed)
    metrics = _run_mpc_on_reference(ref_traj, horizon)
    if metrics is None:
        return "MPC failed to run on uploaded reference. Try a smoother file.", None

    gif_path = "mpc_uploaded_reference.gif"
    create_trajectory_gif(metrics["states"], metrics["ref_interp"], CENTER, WIDTH, [], save_path=gif_path, fps=20)
    report = _format_metrics(
        "Uploaded-Reference MPC Results",
        metrics,
        [
            f"- Input points: {len(points)}",
            "- Accepted formats: x,y or s,offset.",
            _guidance_text(),
        ],
    )
    return report, gif_path


def run_anchor_offset_mpc(start_off, peak_pos_off, peak_neg_off, end_off, speed, horizon):
    anchor_idx = [0, int(np.argmax(CENTER[:, 1])), int(np.argmin(CENTER[:, 1])), len(CENTER) - 1]
    s_anchor = np.asarray(anchor_idx, dtype=float) / float(len(CENTER) - 1)
    offsets = np.asarray([start_off, peak_pos_off, peak_neg_off, end_off], dtype=float)
    offsets = np.clip(offsets, -HALF_WIDTH_MARGIN, HALF_WIDTH_MARGIN)
    coeff = np.polyfit(s_anchor, offsets, 3)
    s_full = np.linspace(0.0, 1.0, len(CENTER))
    curve_offsets = np.polyval(coeff, s_full)
    path = _build_offset_path(curve_offsets)
    ref_traj = _reference_from_xy(path, speed)

    metrics = _run_mpc_on_reference(ref_traj, horizon)
    if metrics is None:
        return "MPC failed to run for anchor offsets. Reduce lateral offsets.", None

    gif_path = "mpc_anchor_offsets.gif"
    create_trajectory_gif(metrics["states"], metrics["ref_interp"], CENTER, WIDTH, [], save_path=gif_path, fps=20)
    report = _format_metrics(
        "Anchor-Offset MPC Results",
        metrics,
        [
            f"- Start offset: {offsets[0]:.2f} m",
            f"- First peak offset: {offsets[1]:.2f} m",
            f"- Negative peak offset: {offsets[2]:.2f} m",
            f"- End offset: {offsets[3]:.2f} m",
            f"- Offset limits enforced: +/-{HALF_WIDTH_MARGIN:.2f} m",
        ],
    )
    return report, gif_path


demo = gr.Blocks(title="MPC Trajectory Tracking for Students")
with demo:
    gr.Markdown("## MPC Trajectory Tracking: Student Lab")
    gr.Markdown(
        "Use structured workflows: upload CSV/XLSX reference points, set guided anchor offsets, or run predefined references."
    )
    gr.Markdown("Set MPC parameters once, then use any input tab.")
    with gr.Row():
        speed = gr.Slider(1.0, 10.0, value=4.0, step=0.1, label="Reference speed (m/s)")
        horizon = gr.Slider(15, 60, value=25, step=5, label="MPC horizon")
    with gr.Accordion("MPC Formulation", open=False):
        gr.Markdown(MPC_EXPLANATION)

    with gr.Tab("Predefined References"):
        gr.Markdown("Run a predefined trajectory with smooth offset profiles and curvature-aware speed.")
        traj_type = gr.Dropdown(
            choices=[
                ("Sinusoidal + Lane Changes", "ref1"),
                ("Zigzag Pattern", "ref2"),
                ("S-Curve Pattern", "ref3"),
            ],
            value="ref1",
            label="Reference type",
        )
        run_reference_btn = gr.Button("Run MPC (Reference)")

    with gr.Tab("Upload Reference"):
        gr.Markdown(
            "Upload a reference file with columns `x,y` or `s,offset`. "
            "Use `s` in [0,1] and keep `offset` within lane limits."
        )
        template_btn = gr.Button("Generate CSV Template")
        template_file = gr.File(label="Template download", interactive=False)
        ref_file = gr.File(label="Upload CSV/XLSX reference", file_types=[".csv", ".xlsx", ".xls"])
        run_upload_btn = gr.Button("Run MPC (Uploaded Reference)")

    with gr.Tab("Guided Anchor Offsets"):
        gr.Markdown(
            "Set lateral offsets at fixed road anchor locations: START, first +peak, first -peak, END. "
            "A cubic offset profile is fit through these anchors."
        )
        start_off = gr.Slider(-HALF_WIDTH_MARGIN, HALF_WIDTH_MARGIN, value=0.0, step=0.05, label="START offset (m)")
        peak_pos_off = gr.Slider(-HALF_WIDTH_MARGIN, HALF_WIDTH_MARGIN, value=1.0, step=0.05, label="First +peak offset (m)")
        peak_neg_off = gr.Slider(-HALF_WIDTH_MARGIN, HALF_WIDTH_MARGIN, value=-1.0, step=0.05, label="First -peak offset (m)")
        end_off = gr.Slider(-HALF_WIDTH_MARGIN, HALF_WIDTH_MARGIN, value=0.0, step=0.05, label="END offset (m)")
        run_anchor_btn = gr.Button("Run MPC (Anchor Offsets)")

    output_text = gr.Textbox(lines=14, label="Results")
    output_gif = gr.Image(type="filepath", label="MPC Animation")

    run_reference_btn.click(run_reference_trajectory, inputs=[traj_type, speed, horizon], outputs=[output_text, output_gif])
    template_btn.click(_create_reference_template, outputs=[template_file])
    run_upload_btn.click(run_uploaded_reference_mpc, inputs=[ref_file, speed, horizon], outputs=[output_text, output_gif])
    run_anchor_btn.click(
        run_anchor_offset_mpc,
        inputs=[start_off, peak_pos_off, peak_neg_off, end_off, speed, horizon],
        outputs=[output_text, output_gif],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
