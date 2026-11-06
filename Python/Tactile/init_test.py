import math
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface

ROBOT_IP = "172.16.34.2"

receive = None
control = None

try:
    receive = RTDEReceiveInterface(ROBOT_IP)
    control = RTDEControlInterface(ROBOT_IP)

    print("Connected to UR5e")

    joint_id = int(input("Enter joint ID (1-6): "))
    angle_deg = float(input("Enter relative angle in degrees: "))

    if joint_id not in range(1, 7):
        raise ValueError("Joint ID must be between 1 and 6")

    current_q = receive.getActualQ()
    target_q = list(current_q)

    target_q[joint_id - 1] += math.radians(angle_deg)

    print("Current joints (rad):", current_q)
    print("Target joints (rad): ", target_q)

    control.moveJ(target_q, 0.1, 0.1)

    print("Move completed")
    print("Final joints (rad):", receive.getActualQ())

except Exception as e:
    print("Error:", e)

finally:
    if control is not None:
        control.stopScript()
        control.disconnect()

    if receive is not None:
        receive.disconnect()