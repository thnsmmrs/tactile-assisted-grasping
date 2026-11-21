"""Headless fixed-observation GoPro ChArUco calibration.

Call calibrate_fixed_camera(cap, num_frames=10) after moving the UR5e
to its observation waypoint. Reuses the existing OpenCV capture handle.
"""
from pathlib import Path
import time
import os

import cv2
import numpy as np

# ChArUco board (measured dimensions, metres)
SQUARES_X, SQUARES_Y = 8, 6
SQUARE_SIZE = 0.0265
MARKER_SIZE = 0.01943
DICT_ID = cv2.aruco.DICT_5X5_50

# Previously measured corners; retain the signs as requested.
# They must be true base-frame XY measurements of these exact corner IDs.
MEASURED_POINTS = {
    34: (0.2875, 0.3470),
    2: (0.3935, 0.4520),
}

CAMERA_DEVICE = "/dev/video0"
WIDTH, HEIGHT, FPS = 1920, 1080, 30
NUM_FRAMES = 30
MIN_CORNERS = 8
CAPTURE_INTERVAL_S = 0.08
MAX_REPROJECTION_RMS_PX = 1.5

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "calibration"
INTRINSICS_FILE = DATA_DIR / "gopro_intrinsics.npz"
OUTPUT_FILE = DATA_DIR / "gopro_fixed_pose.npz"


def open_camera():
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"NV12"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open GoPro capture device {CAMERA_DEVICE}")
    return cap


def get_board():
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
    return cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_SIZE, MARKER_SIZE, dictionary
    )


def estimate_pose(frame, board, detector, K, dist):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = detector.detectBoard(gray)
    # OpenCV can occasionally return an unmatched detection. Never pass
    # mismatched ChArUco arrays to drawing or solvePnP.
    if (corners is None or ids is None
            or len(corners) != len(ids)
            or len(ids) < MIN_CORNERS):
        return None, corners, ids

    ids_1d = ids.flatten().astype(int)
    obj = np.asarray(board.getChessboardCorners()[ids_1d], dtype=np.float64)
    img = np.asarray(corners.reshape(-1, 2), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, corners, ids

    reprojected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum(
        (reprojected.reshape(-1, 2) - img)**2, axis=1
    ))))
    if rms > MAX_REPROJECTION_RMS_PX:
        return None, corners, ids

    R, _ = cv2.Rodrigues(rvec)
    return (R, tvec.reshape(3), rms, len(ids_1d)), corners, ids


def draw_valid_corners(image, corners, ids):
    """Draw ChArUco points ourselves; avoid OpenCV 5.x drawing shape assertions."""
    if corners is None or ids is None:
        return False

    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    corner_ids = np.asarray(ids).reshape(-1)
    if len(points) == 0 or len(points) != len(corner_ids):
        return False

    for (u, v), corner_id in zip(points, corner_ids):
        x, y = int(round(u)), int(round(v))
        cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(image, str(int(corner_id)), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return True


def mean_rotation(rotations):
    U, _, Vt = np.linalg.svd(np.sum(rotations, axis=0))
    D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    return U @ D @ Vt


def summarize_samples(samples, min_frames=10):
    """Reject unstable camera poses and average the remaining rigid transforms."""
    if len(samples) < min_frames:
        raise RuntimeError(f"Only {len(samples)} valid frames; need at least {min_frames}.")

    translations = np.array([s[1] for s in samples])
    middle = np.median(translations, axis=0)
    displacement_mm = np.linalg.norm(translations - middle, axis=1) * 1000
    # Clip large outliers without discarding normal frame-to-frame jitter.
    cutoff_mm = max(5.0, 3.0 * np.median(displacement_mm))
    retained = [s for s, d in zip(samples, displacement_mm) if d <= cutoff_mm]
    if len(retained) < min_frames:
        raise RuntimeError("Camera pose varied too much across captures.")

    R = mean_rotation(np.stack([s[0] for s in retained]))
    t = np.mean(np.stack([s[1] for s in retained]), axis=0)
    translation_scatter_mm = np.sqrt(np.mean(np.sum(
        (np.stack([s[1] for s in retained]) - t)**2, axis=1
    ))) * 1000

    T_camera_board = np.eye(4)
    T_camera_board[:3, :3] = R
    T_camera_board[:3, 3] = t
    return T_camera_board, retained, translation_scatter_mm


def board_to_robot_xy_transform(board):
    """Build 2D rigid board->robot registration from the two measured corners.

    Two points cannot independently resolve a reflected board coordinate frame;
    preserve the same proper-rotation convention as the previous calibration.
    """
    ids = list(MEASURED_POINTS)
    a, b = ids
    board_xy = np.asarray(board.getChessboardCorners(), dtype=float)[:, :2]
    p1, p2 = board_xy[a], board_xy[b]
    q1 = np.asarray(MEASURED_POINTS[a], dtype=float)
    q2 = np.asarray(MEASURED_POINTS[b], dtype=float)
    board_dist = np.linalg.norm(p2 - p1)
    measured_dist = np.linalg.norm(q2 - q1)
    print(f"Board corner separation: {board_dist * 1000:.1f} mm")
    print(f"Robot corner separation: {measured_dist * 1000:.1f} mm")
    if abs(board_dist - measured_dist) > 0.005:
        raise ValueError("Corner measurements disagree with board geometry by >5 mm")

    angle = (np.arctan2(*(q2 - q1)[::-1]) -
             np.arctan2(*(p2 - p1)[::-1]))
    c, s = np.cos(angle), np.sin(angle)
    R_xy = np.array([[c, -s], [s, c]])
    t_xy = q1 - R_xy @ p1
    return R_xy, t_xy


def calibrate_fixed_camera(cap, num_frames=10, max_attempts=None):
    """Recalibrate from an already-open GoPro stream; no UI or robot motion.

    Must be called with UR5e motion complete, board stationary and visible.
    Saves new NPZ atomically only when enough valid frames pass checks.
    Returns a dictionary of calibration results.
    """
    if not cap.isOpened():
        raise RuntimeError("Camera is not open; calibration not changed")
    if num_frames < 3:
        raise ValueError("num_frames must be at least 3")
    if max_attempts is None:
        max_attempts = max(num_frames * 3, 20)
    if max_attempts < num_frames:
        raise ValueError("max_attempts cannot be less than num_frames")

    with np.load(INTRINSICS_FILE) as calib:
        K = np.asarray(calib["K"], dtype=np.float64)
        dist = np.asarray(calib["dist"], dtype=np.float64)
        if int(calib["width"]) != WIDTH or int(calib["height"]) != HEIGHT:
            raise RuntimeError("Saved GoPro intrinsics do not match 1920x1080")

    board = get_board()
    detector = cv2.aruco.CharucoDetector(board)
    R_xy, t_xy = board_to_robot_xy_transform(board)

    samples = []
    attempts = 0
    start = time.perf_counter()
    while len(samples) < num_frames and attempts < max_attempts:
        attempts += 1
        ok, frame = cap.read()
        if not ok:
            # A transient dropped frame should not destroy an existing calibration.
            continue
        if frame.shape != (HEIGHT, WIDTH, 3):
            raise RuntimeError(f"Expected 1080p BGR image, got {frame.shape}")
        pose, _, _ = estimate_pose(frame, board, detector, K, dist)
        if pose is not None:
            samples.append(pose)

    if len(samples) < num_frames:
        raise RuntimeError(
            f"Calibration failed: {len(samples)}/{num_frames} valid images "
            f"in {attempts} attempts. Existing NPZ left untouched."
        )

    # Require sufficient inliers before saving; never overwrite a good file
    # with a failed or noisy calibration.
    min_retained = max(3, int(np.ceil(num_frames * 0.8)))
    T, retained, scatter_mm = summarize_samples(samples, min_frames=min_retained)

    # Direction of board Z depends on OpenCV's board-frame convention.
    camera_origin_board = -T[:3, :3].T @ T[:3, 3]
    if abs(camera_origin_board[2]) < 0.025:
        raise RuntimeError("Camera-board pose inconsistent; NPZ not changed")
    cube_top_z_board = -np.sign(camera_origin_board[2]) * 0.020

    mean_rms = float(np.mean([s[2] for s in retained]))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = OUTPUT_FILE.with_name(OUTPUT_FILE.stem + ".tmp.npz")
    try:
        np.savez(
            temp_file,
            T_camera_board=T,
            K=K,
            dist=dist,
            R_board_to_robot_xy=R_xy,
            t_board_to_robot_xy=t_xy,
            cube_top_z_board=cube_top_z_board,
            board_reference_ids=np.array(list(MEASURED_POINTS)),
            board_reference_robot_xy=np.array(list(MEASURED_POINTS.values())),
            accepted_frames=len(retained),
            mean_reprojection_rms_px=mean_rms,
            translation_scatter_mm=scatter_mm,
        )
        os.replace(temp_file, OUTPUT_FILE)
    finally:
        if temp_file.exists():
            temp_file.unlink()

    duration = time.perf_counter() - start
    print(f"Calibration updated: {len(retained)}/{num_frames} accepted "
          f"({attempts} attempts), RMS={mean_rms:.3f}px, "
          f"scatter={scatter_mm:.2f}mm, elapsed={duration:.2f}s")
    print(f"Saved: {OUTPUT_FILE}")
    return {
        "T_camera_board": T,
        "cube_top_z_board": float(cube_top_z_board),
        "accepted_frames": len(retained),
        "mean_reprojection_rms_px": mean_rms,
        "translation_scatter_mm": float(scatter_mm),
        "elapsed_s": duration,
    }


def main():
    """Optional direct run, no window. Robot must already be at observation."""
    print("UR5e must already be at its saved observation waypoint.")
    print("ChArUco board must be stationary and visible.")
    cap = open_camera()
    try:
        calibrate_fixed_camera(cap, num_frames=10)
    finally:
        cap.release()


if __name__ == "__main__":
    main()