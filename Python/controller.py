import rtde_control
import rtde_io
import numpy as np
import time
import cv2
import socket
import struct
from RobotControl import waypoints


waypoints.go_to_obs()
time.sleep(10)
#waypoints.go_home()