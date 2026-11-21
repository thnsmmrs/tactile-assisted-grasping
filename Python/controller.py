import rtde_control
import rtde_receive
import rtde_io
import numpy as np
import time
import cv2
import socket
import struct
from RobotControl import waypoints, UR5e_IK, grasp_motion
from Vision import object, pixel_conversion, calibration
from concurrent.futures import ThreadPoolExecutor
from RobotControl.gripper import close_gripper
from RobotControl.gripper import open_gripper

ROBOT_IP = "172.16.34.2"

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

#waypoints.go_home(rtde_c)
#time.sleep(2)
#go to observation point
open_gripper()
waypoints.go_to_obs(rtde_c) 

#set up camera and cv output
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"NV12"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("GoPro camera failed to open")

print("Reading first frame...", flush=True)
ret, frame = cap.read()
calibration.calibrate_fixed_camera(cap, num_frames=5)
pixel_conversion.reload_calibration()
print("First frame received:", ret, flush=True)
while True:    
    ret, frame = cap.read()
    if not ret:
        break
    centroid, bounds, box = object.detect_object(frame)


    if centroid is not None:
        u, v = centroid

        X, Y = pixel_conversion.pixel_to_robot(u, v)
        angle = pixel_conversion.box_to_angle(box)
        print(f"Average centroid: ({u:.2f}, {v:.2f})")
        print(f"Robot target: X={X:.4f}, Y={Y:.4f}")
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        cv2.circle(frame, (u, v), 6, (0, 255, 255), -1)
        # Display initial detection and wait for confirmation
        cv2.imshow("Object Detection", frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Object Detection")

        # Execute robot grasp in background thread
        with ThreadPoolExecutor(max_workers=1) as executor:

            future = executor.submit(
                grasp_motion.approach_and_descend,
                rtde_c,
                rtde_r,
                X,
                Y,
                close_gripper=close_gripper,
                angle=angle,
                enable_motion=True
            )

            # Display live tracking while robot moves
            while not future.done():

                ret, live_frame = cap.read()

                if not ret:
                    print("Camera frame unavailable.")
                    break

                live_centroid, live_bounds, live_box = object.detect_object(live_frame)

                if live_centroid is not None:
                    live_u, live_v = live_centroid

                    cv2.drawContours(
                        live_frame,
                        [live_box],
                        0,
                        (0, 255, 0),
                        2
                    )

                    cv2.circle(
                        live_frame,
                        (live_u, live_v),
                        6,
                        (0, 255, 255),
                        -1
                    )


                cv2.imshow("Live Tracking", live_frame)
                cv2.waitKey(1)

            # Retrieve result or raise any error from robot thread
            success = future.result()

        if success:
            cv2.destroyWindow("Live Tracking")
            print("Grasp sequence completed.")
            time.sleep(2)
            open_gripper()
            waypoints.go_to_obs(rtde_c)
            break
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()    
cv2.destroyAllWindows()


#Calibration data
#1044,417 to 838 428
#838, 428 to 841 258
