"""Capture ChArUco calibration images from GoPro HDMI through Elgato HD60 X.

From repo root:
    python Calibration/ChArUco/scripts/capture_gopro.py

SPACE saves a full-resolution clean frame; Q quits.
"""

import time
from pathlib import Path

import cv2

DEVICE = "/dev/video0"
WIDTH, HEIGHT, FPS = 1920, 1080, 30

# Script lives in Calibration/ChArUco/scripts/
SAVE_DIR = Path(__file__).resolve().parents[1] / "data" / "calibration_images"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE = (
    "v4l2src device=/dev/video0 io-mode=mmap ! "
    "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)


def main():
    cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {DEVICE}. Check Elgato connection and GStreamer support.")

    saved = 0
    last_save = 0.0
    cv2.namedWindow("ChArUco HDMI capture", cv2.WINDOW_NORMAL)

    try:
        print("SPACE = save image | Q = quit")
        print(f"Output: {SAVE_DIR}")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame read failed; stopping capture.")
                break

            h, w = frame.shape[:2]
            if (w, h) != (WIDTH, HEIGHT):
                raise RuntimeError(f"Unexpected frame size: {w}x{h}, expected {WIDTH}x{HEIGHT}")

            preview = frame.copy()
            cv2.putText(preview, f"SPACE: save | Q: quit | Saved: {saved}",
                        (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("ChArUco HDMI capture", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == 32 and time.monotonic() - last_save > 0.35:
                path = SAVE_DIR / f"charuco_{time.time_ns()}.png"
                if cv2.imwrite(str(path), frame):
                    saved += 1
                    last_save = time.monotonic()
                    print(f"Saved {saved}: {path.name} ({w}x{h})")
                else:
                    print(f"Failed to save {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Capture ended; saved {saved} images.")


if __name__ == "__main__":
    main()
