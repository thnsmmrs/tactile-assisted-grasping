import rtde_control
import rtde_receive
import rtde_io
import numpy as np
import time
import cv2
import socket
import struct
from RobotControl import waypoints, UR5e_IK, grasp_motion
from Vision import object, pixel_conversion
from RobotControl.gripper import close_gripper
from RobotControl.gripper import open_gripper

ROBOT_IP = "172.16.34.2"

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

waypoints.go_home(rtde_c)
time.sleep(2)
#go to observation point
open_gripper()
waypoints.go_to_obs(rtde_c) 

#set up camera and cv output
pipeline = (
    "v4l2src device=/dev/video0 io-mode=mmap ! "
    "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
print("Opening camera...", flush=True)

if not cap.isOpened():
    raise RuntimeError("Camera failed to open")

print("Reading first frame...", flush=True)
ret, frame = cap.read()
print("First frame received:", ret, flush=True)
while True:    
    ret, frame = cap.read()
    if not ret:
        break
    centroid, bounds = object.detect_object(frame)
    if centroid is not None:
        u, v = centroid

        X, Y = pixel_conversion.pixel_to_robot(u, v)

        print(f"Average centroid: ({u:.2f}, {v:.2f})")
        print(f"Robot target: X={X:.4f}, Y={Y:.4f}")

        success = grasp_motion.approach_and_descend(
            rtde_c,
            rtde_r,
            X,
            Y,
            close_gripper=close_gripper,
            enable_motion=True
        )
        if success:
            print("Grasp sequence completed.")
            time.sleep(2)
            open_gripper()
            waypoints.go_home(rtde_c)
            break
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
cv2.release()

#Calibration data
#1044,417 to 838 428
#838, 428 to 841 258
