import rtde_control
import rtde_io
import numpy as np
import time
import cv2
import socket
import struct
from RobotControl import waypoints
from Vision import object


#go to observation point
waypoints.go_to_obs() 

#set up camera and cv output
pipeline = (
    "v4l2src device=/dev/video0 io-mode=mmap ! "
    "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    object.detect_object(frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
