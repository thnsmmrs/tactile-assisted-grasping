import rtde_control

def go_home(robot):
    robot.moveJ([-2.1e-05, -1.570804, 5e-06, -1.570813, 1e-06, 1.5e-05], 0.9, 0.3)
    
def go_to_obs(robot):
    robot.moveJ([-1.547365, -1.585053, -1.690836, -1.437872, 1.60984, 0.032776], 0.9, 0.3)