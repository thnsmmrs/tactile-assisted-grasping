"""Capture full-resolution GoPro calibration frames for ChArUco.

Place this file at Calibration/ChArUco/scripts/capture_gopro.py.
Run from your project's virtual environment:
    python Calibration/ChArUco/scripts/capture_gopro.py

Space: save current unmodified frame; Q: quit.
"""

from pathlib import Path
import threading
import time

import cv2
import requests

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Python" / "RobotControl"))

from waypoints import go_to_obs, go_home


GOPRO_IP = "172.29.189.51"
BASE_URL = f"http://{GOPRO_IP}:8080"
STREAM_URL = "udp://@0.0.0.0:8554?fifo_size=100000&overrun_nonfatal=1"
SAVE_DIR = Path(__file__).resolve().parents[1] / "data" / "calibration_images"


def camera_command(action):
    response = requests.get(f"{BASE_URL}/gopro/camera/stream/{action}", timeout=5)
    response.raise_for_status()


class LatestFrameCapture:
    """Read frames continuously so the UI doesn't build up a decoder backlog."""

    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open GoPro UDP video stream")
        self.frame = None
        self.received_at = 0.0
        self.lock = threading.Lock()
        self.running = True
        self.failed = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        failures = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                failures += 1
                if failures >= 10:
                    self.failed = True
                    break
                time.sleep(0.05)
                continue
            failures = 0
            with self.lock:
                self.frame = frame
                self.received_at = time.monotonic()
        self.running = False

    def read(self):
        with self.lock:
            if self.frame is None or time.monotonic() - self.received_at > 1.0:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
        if not self.thread.is_alive():
            self.cap.release()
        else:
            print("Warning: video reader is still stopping; exiting process")


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    camera = None
    stream_started = False
    saved_count = 0

    go_to_obs()
    try:
        # Reset a leftover session from another program.
        try:
            camera_command("stop")
        except requests.RequestException as exc:
            print(f"Stream reset warning: {exc}")
        time.sleep(0.5)

        camera_command("start")
        stream_started = True
        camera = LatestFrameCapture(STREAM_URL)

        print(f"Saving images to: {SAVE_DIR}")
        print("SPACE = save image | Q = quit")
        while True:
            if camera.failed:
                raise RuntimeError("GoPro video capture stopped after repeated failures")

            frame = camera.read()
            if frame is not None:
                # Only the preview gets annotations; saved images remain untouched.
                preview = frame.copy()
                cv2.putText(preview, "SPACE: save  |  Q: quit", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("ChArUco calibration capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
                go_home()
            if key == 32:
                if frame is None:
                    print("No fresh frame available; not saving")
                    continue
                filepath = SAVE_DIR / f"charuco_{time.time_ns()}.png"
                if not cv2.imwrite(str(filepath), frame):
                    print(f"ERROR saving {filepath}")
                    continue
                saved_count += 1
                print(f"Saved #{saved_count}: {filepath.name} ({frame.shape[1]}x{frame.shape[0]})")
            time.sleep(0.005)

    finally:
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()
        if stream_started:
            try:
                camera_command("stop")
            except requests.RequestException as exc:
                print(f"GoPro stop warning: {exc}")
        print(f"Captured {saved_count} images")


if __name__ == "__main__":
    main()
