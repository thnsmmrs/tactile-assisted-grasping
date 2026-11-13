import numpy as np

REFERENCE_PIXEL = np.array([836.0, 429.0])

REFERENCE_ROBOT = np.array([
    0.019604384623022452,
    -0.7270748846840315
])

PIXEL_AXES = np.array([
    [-206.0, -3.0],
    [11.0, 170.0]
])

X_OFFSET = -0.008
Y_OFFSET = 0.016

def pixel_to_robot(u, v):
    pixel_offset = np.array([u, v]) - REFERENCE_PIXEL

    movement = np.linalg.solve(PIXEL_AXES, pixel_offset)

    robot_xy = REFERENCE_ROBOT + 0.2 * movement

    robot_xy[0] += X_OFFSET
    robot_xy[1] += Y_OFFSET

    return float(robot_xy[0]), float(robot_xy[1])