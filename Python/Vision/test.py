import cv2

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"NV12"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Opened:", cap.isOpened(), flush=True)
print("Resolution:",
      cap.get(cv2.CAP_PROP_FRAME_WIDTH),
      cap.get(cv2.CAP_PROP_FRAME_HEIGHT), flush=True)

print("Reading...", flush=True)
ret, frame = cap.read()
print("Success:", ret, flush=True)

if ret:
    print("Shape:", frame.shape)
    cv2.imshow("GoPro Test", frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()