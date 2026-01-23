DT = 0.05
HORIZON = 60  # 3 seconds at 0.05s dt
SIM_STEPS = 400

VEHICLE = dict(
    m=3.0,
    Iz=0.05,
    lf=0.15,
    lr=0.15,
    Cf=35.0,
    Cr=35.0
)

LIMITS = dict(
    steer_max=0.5,
    steer_rate=3.0,
    accel_max=2.0
)

NOISE = dict(
    yaw_rate=0.02,
    speed=0.05,
    pos=0.02
)

# MPC weights
Q_TRACK = [40, 40, 10, 5, 1, 1]
R_INPUT = [3.0, 1.0]
R_JERK = [20.0, 5.0]
PROGRESS_WEIGHT = 15.0

# Reference speed for trajectory tracking
REFERENCE_SPEED = 4.0
