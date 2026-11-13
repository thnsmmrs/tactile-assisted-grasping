import cv2
from object import detect_object

pipeline = (
    "v4l2src device=/dev/video0 io-mode=mmap ! "
    "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise RuntimeError("Camera failed to open")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Detection region
        x1, x2 = int(w * 0.3), int(w * 0.65)
        y1, y2 = int(h * 0.1), int(h * 0.7)

        # Detect cube using existing function
        centroid, bounds = detect_object(frame)

        # Display ROI boundary (yellow)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if centroid is not None:
            u, v = centroid
            x, y, bw, bh = bounds

            # Centroid (red)
            cv2.circle(frame, (u, v), 8, (0, 0, 255), -1)

            # Bounding box (green)
            cv2.rectangle(
                frame, (x, y), (x + bw, y + bh),
                (0, 255, 0), 2
            )

            cv2.putText(
                frame, f"Centroid: ({u}, {v})",
                (u + 15, v - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

        # Show HSV mask using the same thresholds
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (95, 70, 50), (135, 255, 255))

        cv2.imshow("Cube Detection", frame)
        cv2.imshow("Blue Mask", mask)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()