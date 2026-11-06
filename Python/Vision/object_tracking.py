
import cv2
import numpy as np
import requests
import threading
import time

GOPRO_IP = "172.29.189.51"
BASE_URL = f"http://{GOPRO_IP}:8080"

STREAM_URL = (
    "udp://@0.0.0.0:8554"
    "?fifo_size=100000&overrun_nonfatal=1"
)

LOWER_BLUE = np.array([95, 70, 50])
UPPER_BLUE = np.array([135, 255, 255])
MIN_CONTOUR_AREA = 100

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def camera_command(action):
    url = f"{BASE_URL}/gopro/camera/stream/{action}"
    response = requests.get(url, timeout=5)
    print(f"GoPro {action}: HTTP {response.status_code}")
    response.raise_for_status()


class LatestFrameCapture:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open GoPro stream.")

        self.frame = None
        self.timestamp = 0.0
        self.lock = threading.Lock()
        self.running = True
        self.failed = False

        self.thread = threading.Thread(
            target=self.update, daemon=True
        )
        self.thread.start()

    def update(self):
        failures = 0

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                failures += 1
                if failures >= 10:
                    self.failed = True
                    break
                time.sleep(0.05)
                continue

            failures = 0

            frame = cv2.resize(
                frame, (FRAME_WIDTH, FRAME_HEIGHT)
            )

            with self.lock:
                self.frame = frame
                self.timestamp = time.monotonic()

        self.running = False

    def read(self):
        with self.lock:
            if self.frame is None:
                return None

            # Avoid reusing a frame when capture has stalled.
            if time.monotonic() - self.timestamp > 1.0:
                return None

            return self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

        if not self.thread.is_alive():
            self.cap.release()
        else:
            print("Warning: camera thread still active.")


class CubeCamera:
    """Starts/stops GoPro and exposes latest camera frame."""

    def __init__(self):
        self.capture = None
        self.started = False

    def start(self):
        try:
            camera_command("stop")
        except requests.RequestException as e:
            print("Reset warning:", e)

        time.sleep(0.5)

        camera_command("start")
        self.started = True

        try:
            self.capture = LatestFrameCapture(STREAM_URL)
        except Exception:
            self.stop()
            raise

    def read(self):
        if self.capture is None:
            return None

        if self.capture.failed:
            raise RuntimeError("GoPro video capture failed.")

        return self.capture.read()

    def stop(self):
        if self.capture is not None:
            self.capture.stop()
            self.capture = None

        cv2.destroyAllWindows()

        if self.started:
            try:
                camera_command("stop")
            except requests.RequestException as e:
                print("GoPro stop warning:", e)

        self.started = False


def detect_blue_cube(frame, return_center=False):
    """Draw detection overlay and optionally return (u, v)."""

    height, width = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv, LOWER_BLUE, UPPER_BLUE
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_center = (width // 2, height // 2)
    center = None

    cv2.drawMarker(
        frame, image_center,
        (255, 255, 255),
        cv2.MARKER_CROSS, 25, 2
    )

    if contours:
        largest = max(
            contours, key=cv2.contourArea
        )

        if cv2.contourArea(largest) > MIN_CONTOUR_AREA:
            moments = cv2.moments(largest)

            if moments["m00"] != 0:
                cx = int(
                    moments["m10"] / moments["m00"]
                )
                cy = int(
                    moments["m01"] / moments["m00"]
                )

                center = (cx, cy)

                x, y, w, h = cv2.boundingRect(largest)

                cv2.rectangle(
                    frame, (x, y),
                    (x + w, y + h),
                    (0, 255, 0), 2
                )

                cv2.circle(
                    frame, center, 5,
                    (0, 0, 255), -1
                )

                dx = cx - image_center[0]
                dy = cy - image_center[1]

                cv2.putText(
                    frame,
                    f"Cube: ({cx}, {cy})",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"Offset: ({dx}, {dy}) px",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

    if return_center:
        return frame, center

    return frame


def main():
    camera = CubeCamera()

    try:
        camera.start()
        print("Tracking blue cube. Press Q to quit.")

        while True:
            frame = camera.read()

            if frame is not None:
                processed = detect_blue_cube(frame)

                cv2.imshow(
                    "GoPro - Blue Cube Tracking",
                    processed
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.005)

    finally:
        camera.stop()


if __name__ == "__main__":
    main()
