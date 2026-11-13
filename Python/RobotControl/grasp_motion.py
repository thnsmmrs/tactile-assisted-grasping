
import numpy as np
from RobotControl import UR5e_IK
from RobotControl.gripper import close_gripper

# Manually taught pose with 30 mm cube centered between fingers
GRASP_REFERENCE = [
    0.019604384623022452,
    -0.7270748846840315,
    -0.010923665669001409,
    0.15036606486333268,
    -3.11174312219261,
    -0.08777831443041605
]

# Candidate: 50 mm above taught grasp position.
# Must be checked for clearance before enabling motion.
APPROACH_Z = GRASP_REFERENCE[2] + 0.050

# Restrict initial tests to within 30 mm of taught reference.
MAX_XY_OFFSET = 1.00

AUTO_MODE = True

def approach_and_descend(rtde_c, rtde_r, X, Y, close_gripper,enable_motion=False):
    """
    Approach above the cube, then descend to taught grasp Z.

    X, Y: robot base-frame coordinates in meters
    enable_motion: False = calculate/print only
    """

    # Restrict initial testing to the taught grasp area
    distance = np.hypot(
        X - GRASP_REFERENCE[0],
        Y - GRASP_REFERENCE[1]
    )

    if not np.isfinite(distance) or distance > MAX_XY_OFFSET:
        print("Target outside initial test region.")
        return False

    # Build grasp pose using detected XY
    GRASP_Z_OFFSET = -0.002  # 2 mm lower

    grasp_pose = GRASP_REFERENCE.copy()
    grasp_pose[0] = X
    grasp_pose[1] = Y
    grasp_pose[2] += GRASP_Z_OFFSET

    # Build elevated approach pose
    approach_pose = grasp_pose.copy()
    approach_pose[2] = APPROACH_Z

    # Check that both poses have IK solutions
    approach_joints = UR5e_IK.calculate_ik(
        rtde_c, rtde_r, approach_pose
    )

    grasp_joints = UR5e_IK.calculate_ik(
        rtde_c, rtde_r, grasp_pose
    )

    if approach_joints is None or grasp_joints is None:
        print("IK failed. No movement commanded.")
        return False

    print("\nApproach TCP:", np.round(approach_pose, 4))
    print("Grasp TCP:   ", np.round(grasp_pose, 4))
    print("Approach joints (deg):",
          np.round(np.degrees(approach_joints), 2))
    print("Grasp joints (deg):",
          np.round(np.degrees(grasp_joints), 2))

    if not enable_motion:
        print("DRY RUN: No movement commanded.")
        return True

    if not AUTO_MODE:
        confirmation = input("Type APPROACH to execute moveJ: ")
        if confirmation != "APPROACH":
            return False

    # Stage 1: Joint-space movement to elevated position
    rtde_c.moveJ(approach_joints, 0.6, 0.1)
    print("Approach movement completed.")

    if not AUTO_MODE:
        confirmation = input("Type DESCEND to execute moveL: ")
        if confirmation != "DESCEND":
            print("Descent cancelled.")
            return False

    rtde_c.moveL(grasp_pose, 0.3, 0.05)
    print("Descent completed.")

    if not AUTO_MODE:
        confirmation = input("Type GRASP to close gripper: ")
        if confirmation != "GRASP":
            print("Grasp cancelled.")
            return False

    status = close_gripper()

    if status not in ("OBJ 2", "OBJ 3"):
        print(f"Unexpected gripper status: {status}")
        return False

    if not AUTO_MODE:
        confirmation = input("Verify grip, then type LIFT: ")
        if confirmation != "LIFT":
            print("Lift cancelled.")
            return False

    #rtde_c.moveL(approach_pose, 0.02, 0.05)

    print("Lift completed. Holding object.")
    return True