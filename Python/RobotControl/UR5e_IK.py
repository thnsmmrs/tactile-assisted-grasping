def calculate_ik(rtde_c, rtde_r, target_pose):
    current_joints = rtde_r.getActualQ()

    try:
        joint_angles = rtde_c.getInverseKinematics(
            target_pose,
            current_joints
        )

        return joint_angles

    except RuntimeError as e:
        print(f"IK calculation failed: {e}")
        return None