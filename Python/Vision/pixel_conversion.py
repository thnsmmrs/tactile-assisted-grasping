"""GoPro pixel to robot XY for a fixed UR5e observation pose."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_FILE = ROOT / "data" / "calibration" / "gopro_fixed_pose.npz"


def reload_calibration():
    """Refresh in-memory transforms after calibration overwrites the NPZ."""
    global K, dist, T_board_camera, R_board_to_robot, t_board_to_robot, CUBE_TOP_Z
    with np.load(CALIBRATION_FILE) as data:
        K = np.array(data["K"], dtype=float)
        dist = np.array(data["dist"], dtype=float)
        T_board_camera = np.linalg.inv(np.array(data["T_camera_board"], dtype=float))
        R_board_to_robot = np.array(data["R_board_to_robot_xy"], dtype=float)
        t_board_to_robot = np.array(data["t_board_to_robot_xy"], dtype=float)
        CUBE_TOP_Z = float(data["cube_top_z_board"])


def pixel_to_robot(u, v):
    pixel = np.array([[[u, v]]], dtype=np.float64)
    norm = cv2.undistortPoints(pixel, K, dist)[0, 0]
    ray_camera = np.array([norm[0], norm[1], 1.0])

    camera_origin_board = T_board_camera[:3, 3]
    ray_board = T_board_camera[:3, :3] @ ray_camera
    if abs(ray_board[2]) < 1e-9:
        raise ValueError("Ray is parallel to cube-top plane")
    scale = (CUBE_TOP_Z - camera_origin_board[2]) / ray_board[2]
    if scale <= 0:
        raise ValueError("Cube-top plane is behind camera")

    board_xy = (camera_origin_board + scale * ray_board)[:2]
    robot_xy = R_board_to_robot @ board_xy + t_board_to_robot
    return float(robot_xy[0]+0.015), -1*float(robot_xy[1])


def box_to_angle(box):
    """Return cube orientation in robot XY, modulo 90 degrees."""
    box = np.asarray(box, dtype=float).reshape(4, 2)
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    i = int(np.argmax(lengths))
    p1 = box[i]
    p2 = box[(i + 1) % 4]
    x1, y1 = pixel_to_robot(*p1)
    x2, y2 = pixel_to_robot(*p2)
    angle = np.arctan2(y2 - y1, x2 - x1)
    return float((angle + np.pi / 4) % (np.pi / 2) - np.pi / 4)


reload_calibration()
