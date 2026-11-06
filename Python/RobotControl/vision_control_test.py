
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rtde_receive

# Allow imports from Python/Vision/
VISION_DIR = Path(__file__).resolve().parents[1] / "Vision"
sys.path.insert(0, str(VISION_DIR))

from object_tracking import CubeCamera, detect_blue_cube
from gripper_test import Gripper

ROBOT_IP = "172.16.34.2"

# --------------- WAYPOINTS ----------------

OBSERVATION_Q = [
    -1.413619, -1.599147, -1.889122,
    -1.189968, 1.491240, 0.057750
]

OBSERVATION_TCP = [
    -0.075841, -0.507088, 0.228571,
    0.153413, -3.054286, 0.066694
]

# Manually taught TCP pose for grasping the cube.
# Finger tips must clear the table.
GRASP_TCP = None

# Base-frame Z of a verified collision-free approach.
SAFE_Z = None

# ------------ VISION CALIBRATION ------------

# Each sample is:
# (cube_pixel_u, cube_pixel_v, robot_x, robot_y)
#
# Use at least four points spread around the workspace.
# All images must use the same observation pose.
CALIBRATION = [
    # (u1, v1, x1, y1),
    # (u2, v2, x2, y2),
    # (u3, v3, x3, y3),
    # (u4, v4, x4, y4),
]

# --------------------------------------------

SPEED = 0.03
ACCELERATION = 0.08

# Maximum permitted calibration fit RMS (meters).
MAX_CALIBRATION_RMS = 0.005

# Maximum allowable observation pose difference.
POSITION_TOLERANCE = 0.005
ORIENTATION_TOLERANCE = 0.05


def teach():
    """Print current robot joint and TCP coordinates."""
    robot = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

    try:
        q = list(robot.getActualQ())
        tcp = list(robot.getActualTCPPose())

        print("\nCopy these into your script:")
        print("OBSERVATION_Q =", [round(v, 6) for v in q])
        print("OBSERVATION_TCP =", [round(v, 6) for v in tcp])

        print("\nCurrent TCP position (meters):")
        print("X =", tcp[0])
        print("Y =", tcp[1])
        print("Z =", tcp[2])

    finally:
        robot.disconnect()


def verify_observation_pose():
    """Confirm robot remains at calibrated camera pose."""
    if OBSERVATION_TCP is None:
        raise ValueError("Set OBSERVATION_TCP first.")

    if len(OBSERVATION_TCP) != 6:
        raise ValueError("Invalid observation TCP pose.")

    robot = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

    try:
        current = np.array(
            robot.getActualTCPPose(), dtype=float
        )
    finally:
        robot.disconnect()

    expected = np.array(OBSERVATION_TCP, dtype=float)

    pos_error = np.linalg.norm(
        current[:3] - expected[:3]
    )
    rot_error = np.linalg.norm(
        current[3:] - expected[3:]
    )

    if pos_error > POSITION_TOLERANCE:
        raise RuntimeError(
            "Robot isn't at observation position."
        )

    if rot_error > ORIENTATION_TOLERANCE:
        raise RuntimeError(
            "Robot orientation differs from observation."
        )


def get_calibration():
    """Fit pixel (u,v) to robot base-frame (x,y)."""

    points = np.asarray(CALIBRATION, dtype=float)

    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(
            "Calibration must contain (u,v,x,y) rows."
        )

    if len(points) < 4:
        raise ValueError(
            "Collect at least four calibration points."
        )

    pixels = points[:, :2]
    robot_xy = points[:, 2:]

    A = np.column_stack([
        pixels[:, 0],
        pixels[:, 1],
        np.ones(len(pixels))
    ])

    coefficients, _, rank, _ = np.linalg.lstsq(
        A, robot_xy, rcond=None
    )

    if rank != 3:
        raise ValueError(
            "Calibration points are degenerate."
        )

    predicted = A @ coefficients

    errors = np.linalg.norm(
        predicted - robot_xy, axis=1
    )

    rms = np.sqrt(np.mean(errors ** 2))

    print(f"Calibration RMS: {rms * 1000:.2f} mm")

    if rms > MAX_CALIBRATION_RMS:
        raise RuntimeError(
            "Calibration error too large. Recalibrate."
        )

    hull = cv2.convexHull(
        pixels.astype(np.float32)
    )

    return coefficients, hull


def image_to_robot(center, coefficients, hull):
    """Reject uncalibrated targets and calculate XY."""

    u, v = center

    if cv2.pointPolygonTest(
        hull, (float(u), float(v)), False
    ) < 0:
        raise RuntimeError(
            "Cube is outside the calibrated image region."
        )

    x, y = np.array(
        [u, v, 1.0]
    ) @ coefficients

    if not np.isfinite([x, y]).all():
        raise RuntimeError("Invalid target coordinates.")

    return float(x), float(y)


def observe_cube():
    """Preview video, select cube with SPACE."""
    camera = CubeCamera()

    try:
        camera.start()

        print("\nSPACE: capture cube center")
        print("Q: cancel")

        while True:
            frame = camera.read()

            if frame is not None:
                frame, center = detect_blue_cube(
                    frame, return_center=True
                )

                cv2.imshow("Vision Control", frame)
            else:
                center = None

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                return None

            if key == 32:
                if center is not None:
                    print("Cube pixel center:", center)
                    return center

                print("Cube not detected.")

            time.sleep(0.005)

    finally:
        camera.stop()


def get_target():
    """Locate cube and estimate robot XY."""
    coefficients, hull = get_calibration()

    center = observe_cube()

    if center is None:
        return None

    x, y = image_to_robot(
        center, coefficients, hull
    )

    print("\nVision target:")
    print(f"Pixels: {center}")
    print(f"Robot X: {x:.5f} m")
    print(f"Robot Y: {y:.5f} m")

    return x, y


def calculate_poses(x, y):
    """Construct grasp and overhead approach poses."""

    if GRASP_TCP is None or SAFE_Z is None:
        raise ValueError(
            "Set GRASP_TCP and SAFE_Z first."
        )

    if len(GRASP_TCP) != 6:
        raise ValueError("Invalid GRASP_TCP.")

    grasp = list(GRASP_TCP)

    grasp[0] = x
    grasp[1] = y

    approach = grasp.copy()
    approach[2] = SAFE_Z

    if SAFE_Z <= grasp[2] + 0.03:
        raise ValueError(
            "Approach must be at least 30 mm "
            "above grasp height."
        )

    return approach, grasp


def move_robot(target, control, speed=SPEED):
    result = control.moveL(
        target, speed, ACCELERATION
    )

    if not result:
        raise RuntimeError("Robot motion failed.")


def execute_motion(x, y, mode):
    """Run supervised approach or pick sequence."""
    import rtde_control

    approach, grasp = calculate_poses(x, y)

    print("\nPredicted approach pose:")
    print(approach)

    print("\nPredicted grasp pose:")
    print(grasp)

    print(
        "\nWARNING: This script does not check "
        "robot collisions, joint limits or table geometry."
    )
    print(
        "Verify the full route, mounted TCP, "
        "gripper clearance and available E-stop."
    )

    confirmation = (
        "APPROACH" if mode == "approach" else "PICK"
    )

    if input(
        f"\nType {confirmation} to continue: "
    ) != confirmation:
        print("Motion cancelled.")
        return

    # Observation pose must still be unchanged.
    verify_observation_pose()

    control = None

    try:
        # Opening the gripper before approaching prevents
        # descending toward the object with closed fingers.
        with Gripper() as gripper:
            gripper.open()

            control = rtde_control.RTDEControlInterface(
                ROBOT_IP
            )

            print("Moving above target...")
            move_robot(approach, control)

            if mode == "approach":
                print(
                    "Approach complete. "
                    "No descent or grasp executed."
                )
                return

            print("Descending to taught grasp height...")
            move_robot(grasp, control, speed=0.015)

            print("Closing gripper...")
            result = gripper.close()

            print("Gripper result:", result)

            # OBJ 2 generally indicates contact while closing.
            # Do not automatically lift on an unconfirmed grasp.
            if result != "OBJ 2":
                print(
                    "Grasp not confirmed. "
                    "Robot remains at grasp pose."
                )
                return

            print("Lifting cube...")
            move_robot(approach, control)

            print("\nPickup sequence completed.")

    finally:
        if control is not None:
            control.stopScript()
            control.disconnect()


def main():
    mode = (
        sys.argv[1].lower()
        if len(sys.argv) > 1 else "teach"
    )

    if mode == "teach":
        teach()
        return

    if mode not in (
        "observe", "preview", "approach", "pick"
    ):
        print(
            "Usage: python vision_control_test.py "
            "[teach|observe|preview|approach|pick]"
        )
        return

    verify_observation_pose()

    if mode == "observe":
        observe_cube()
        return

    target = get_target()

    if target is None:
        print("No cube selected.")
        return

    if mode == "preview":
        print("\nPreview only. No robot movement.")
        return

    execute_motion(*target, mode)


if __name__ == "__main__":
    main()
