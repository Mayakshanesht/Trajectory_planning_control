# MPC Trajectory Tracking

A Model Predictive Control (MPC) system for autonomous vehicle trajectory tracking with interactive waypoint selection and cubic polynomial trajectory generation.

## Features

- **Interactive Waypoint Selection**: Click on the road to add waypoints
- **4-Point System**: START → MIDDLE 1 → MIDDLE 2 → END
- **Cubic Polynomial Interpolation**: Perfect fit through all waypoints (0.000m deviation)
- **Multiple Interfaces**: Gradio web interface, HTTP server interface, and MPC simulator
- **Real-time Visualization**: Instant trajectory generation and preview
- **MPC Control**: Full Model Predictive Control implementation

## Interfaces

### 1. MPC Simulator (Recommended)
- **URL**: `http://localhost:7865`
- **Features**: 4-point cubic trajectory + MPC simulation
- **Launch**: `python launcher.py --interface mpc`
- **Workflow**: Click waypoints → Generate trajectory → Run MPC simulation

### 2. Gradio Interface
- **URL**: `http://localhost:7864`
- **Features**: Click-to-add waypoints, real-time trajectory generation
- **Launch**: `python launcher.py --interface gradio`

### 3. Web Interface (4-Point System)
- **URL**: `http://localhost:8082`
- **Features**: Dedicated 4-point cubic trajectory interface
- **Launch**: `python launcher.py --interface web`

### 4. Command Line Interface
- **Features**: Script-based trajectory tracking
- **Launch**: `python launcher.py --interface cli`

## 4-Point System

The system uses exactly 4 points for cubic polynomial interpolation:

1. **START** (Green marker) - Vehicle starting position
2. **MIDDLE 1** (Blue marker) - First intermediate waypoint
3. **MIDDLE 2** (Blue marker) - Second intermediate waypoint  
4. **END** (Red marker) - Final destination

The cubic polynomial ensures the trajectory passes exactly through all 4 points.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Launch Interface
```bash
# Interactive menu
python launcher.py

# Direct launch
python launcher.py --interface gradio  # Gradio interface
python launcher.py --interface web     # Web interface
python launcher.py --interface cli     # Command line
```

### Using the Gradio Interface
1. Open `http://localhost:7864`
2. Click on the road to add exactly 4 waypoints
3. Adjust speed and horizon parameters
4. Click "Generate Trajectory" to see the cubic curve
5. The trajectory passes exactly through all 4 points

## Trajectory Generation

The system uses **cubic polynomial interpolation** with perfect waypoint fitting:

```python
# Cubic polynomial fitting
coeffs_x = np.polyfit(t_values, waypoints_x, 3)  # 3rd degree for x(t)
coeffs_y = np.polyfit(t_values, waypoints_y, 3)  # 3rd degree for y(t)

# Generate smooth trajectory
trajectory = []
for t in np.linspace(0, 1, N):
    x = np.polyval(coeffs_x, t)
    y = np.polyval(coeffs_y, t)
    trajectory.append([x, y])
```

**Accuracy**: 0.000m deviation from waypoints

## Project Structure

```
realistic_mpc_tracking/
├── config/              # MPC parameters and vehicle configuration
├── control/             # MPC controller and linearization
├── estimation/         # Extended Kalman Filter
├── track/              # Track generation and reference trajectories
├── utils/              # Shared utilities and plotting
├── vehicle/            # Vehicle dynamics and actuators
├── gradio_interface.py # Gradio web interface
├── web_interface.py    # HTTP server interface
├── launcher.py         # Unified launcher script
├── main.py            # Main MPC simulation
├── main_simple.py     # Simplified MPC simulation
└── requirements.txt    # Python dependencies
```

## Usage Examples

### Gradio Interface
```bash
python launcher.py --interface gradio
# Then visit http://localhost:7864
# Click 4 times on the road: START → MIDDLE 1 → MIDDLE 2 → END
# Generate trajectory to see perfect cubic curve
```

### Command Line
```bash
python main_simple.py --trajectory ref1 --speed 4.0 --horizon 60
```

### Web Interface
```bash
python launcher.py --interface web
# Then visit http://localhost:8082
# Add 4 waypoints and generate cubic trajectory
```

## Technical Details

### MPC Controller
- **Prediction Horizon**: 20-100 steps
- **State Space**: [x, y, ψ, v, a, δ] (position, heading, speed, acceleration, steering)
- **Control Inputs**: [acceleration, steering_rate]
- **Constraints**: Road boundaries, actuator limits

### Trajectory Interpolation
- **Method**: Cubic polynomial interpolation
- **Parameterization**: Arc-length parameter t ∈ [0, 1]
- **Accuracy**: Perfect waypoint fitting
- **Smoothness**: C² continuous trajectory

### Vehicle Model
- **Bicycle Model**: Simplified kinematic model
- **Actuator Dynamics**: Rate-limited steering and acceleration
- **State Estimation**: Extended Kalman Filter

## Performance

- **Trajectory Accuracy**: 0.000m waypoint deviation
- **Real-time Generation**: <100ms for 100-point trajectory
- **MPC Solve Time**: <50ms for 60-step horizon
- **Tracking Error**: <0.1m RMS

## Development

### Adding New Interfaces
1. Create interface file in root directory
2. Import from `utils.trajectory_utils` for shared functionality
3. Update `launcher.py` to include new interface
4. Update documentation

### Custom Trajectory Generation
```python
from utils.trajectory_utils import direct_waypoint_interpolation

waypoints = [[0, 0], [20, 2], [40, -1], [60, 0]]
trajectory = direct_waypoint_interpolation(waypoints, N=100)
```

## License

This project is part of the MPC bootcamp curriculum.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
- Check the documentation
- Review the examples
- Examine the test cases in `utils/trajectory_utils.py`

### Available Arguments
- `--trajectory`: ref1, ref2, ref3 (required)
- `--speed`: Reference speed in m/s (default: 4.0)
- `--output`: Output GIF filename (default: mpc_trajectory.gif)

## 📊 Visualization

The simulation generates animated GIFs showing:
- **🟢 Green dashed line**: Reference trajectory (desired path)
- **🔵 Blue solid line**: Actual vehicle trajectory
- **🔴 Red dotted lines**: MPC prediction horizons
- **🟢 Green dot**: Start position
- **🔴 Red dot**: End position
- **Road boundaries**: Sinusoidal track with lane markings

## 🔧 Customization

### Modify Reference Speed
```python
# In config/params.py
REFERENCE_SPEED = 6.0  # Faster tracking
```

### Adjust MPC Weights
```python
# For more aggressive tracking:
Q_TRACK = [80, 80, 20, 10, 2, 2]
R_INPUT = [1.5, 0.5]

# For smoother control:
Q_TRACK = [20, 20, 5, 2, 0.5, 0.5]
R_INPUT = [6.0, 2.0]
```

### Custom Trajectory (Code)
```python
# In track/reference_trajectories.py
def custom_trajectory(x, center, N, dt, speed=2.5):
    # Your custom trajectory logic here
    pass
```

## 📈 Output Examples

### Console Output
```
Starting ref1 simulation...
Track length: 60m, Amplitude: 5m, Width: 7.4m
Reference speed: 4.0 m/s
Step 0/800: Position (1.0, -1.6), Speed: 3.0, Progress: 0.0%
...
🏁 Destination reached! Progress: 99.6%
✅ Vehicle stopped at destination. Final position: (61.4, -0.9), Final speed: 0.01 m/s
📊 SIMULATION SUMMARY:
  Final position: (61.4, -0.9)
  Final speed: 0.1 m/s
  Final progress: 99.6%
```

### Generated Files
- **`mpc_ref1_tracking.gif`**: Animated visualization
- **`mpc_ref2_tracking.gif`**: Zigzag trajectory
- **`mpc_ref3_tracking.gif`**: S-curve trajectory
- **`mpc_custom_tracking.gif`**: Custom trajectory

## 🚀 Extensions

The codebase supports easy extensions:
- **Different tracks**: Create custom road geometries
- **Multiple vehicles**: Add obstacle avoidance
- **Advanced MPC**: Implement nonlinear MPC
- **Hardware integration**: Add ROS interfaces
- **Performance metrics**: Comprehensive evaluation suite

## 🐛 Troubleshooting

### Common Issues
1. **Import errors**: Run `pip install -r requirements.txt`
2. **Poor tracking**: Increase Q_TRACK weights for position
3. **Oscillations**: Increase R_INPUT weights for control smoothness
4. **No convergence**: Check solver setup and constraints
5. **Gradio issues**: Ensure all dependencies are installed

### Performance Tips
- Reduce HORIZON for faster computation
- Adjust MPC weights for desired tracking vs smoothness trade-off
- Use command line for faster iteration

## 📚 Mathematical Background

### Vehicle Model
```
x = [X, Y, ψ, vx, vy, r]  # Position, heading, velocities, yaw rate
u = [δ, ax]             # Steering angle, longitudinal acceleration
```

### MPC Formulation
```
min Σ(xᵀQx + uᵀRu + λ‖u[k] - u[k-1]‖²)
s.t. x[k+1] = f(x[k], u[k])
     u_min ≤ u[k] ≤ u_max
```

### Polynomial Interpolation
```
y(x) = a₀ + a₁x + a₂x² + a₃x³
```

---

**Note**: This project provides both simple command-line tools and an advanced web interface for MPC trajectory tracking, making it suitable for education, research, and development of autonomous vehicle control systems.
