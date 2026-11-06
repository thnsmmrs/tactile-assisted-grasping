import rtde_control

def go_home():
    robot = rtde_control.RTDEControlInterface("172.16.34.2")
    robot.moveJ([-2.1e-05, -1.570804, 5e-06, -1.570813, 1e-06, 1.5e-05], 0.2, 0.3)
    robot.disconnect()
    
def go_to_obs():
    robot = rtde_control.RTDEControlInterface("172.16.34.2")
    robot.moveJ([-1.547365, -1.585053, -1.690836, -1.437872, 1.60984, 0.032776], 0.2, 0.3)
    robot.disconnect()