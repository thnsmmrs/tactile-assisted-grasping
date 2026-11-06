import rtde_control

def go_to_home():
    robot = rtde_control.RTDEControlInterface("172.16.34.2")
    robot.moveJ([-2.1e-05, -1.570804, 5e-06, -1.570813, 1e-06, 1.5e-05], 0.2, 0.3)
    robot.disconnect()