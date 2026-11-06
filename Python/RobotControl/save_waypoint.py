
import rtde_receive

ROBOT_IP = "172.16.34.2"

robot = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

try:
    joints = robot.getActualQ()
    tcp = robot.getActualTCPPose()

    print("\nJOINT WAYPOINT (radians):")
    print([round(q, 6) for q in joints])

    print("\nTCP WAYPOINT (meters + rotation vector):")
    print([round(p, 6) for p in tcp])

finally:
    robot.disconnect()
