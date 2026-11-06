
import os
import time

# Configure FFmpeg before importing OpenCV
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "fflags;nobuffer|flags;low_delay"
)

import cv2
import requests

GOPRO_IP = "172.29.189.51"
BASE_URL = f"http://{GOPRO_IP}:8080"
STREAM_URL = "udp://@0.0.0.0:8554"


def camera_command(command):
    url = f"{BASE_URL}/gopro/camera/stream/{command}"
    response = requests.get(url, timeout=5)
    print(f"{command}: HTTP {response.status_code}")
    print(response.text)
    response.raise_for_status()


def main():
    cap = None

    try:
        # Reset any previous stream session
        camera_command("stop")
        time.sleep(1)

        print("Starting GoPro preview stream...")
        camera_command("start")

        print("Opening stream in OpenCV...")
        cap = cv2.VideoCapture(
            STREAM_URL, cv2.CAP_FFMPEG
        )

        if not cap.isOpened():
            print("Could not open video stream.")
            return

        previous_time = time.monotonic()

        while True:
            success, frame = cap.read()

            if not success:
                print("Frame unavailable. Stream may have stopped.")
                break

            current_time = time.monotonic()
            dt = current_time - previous_time
            previous_time = current_time

            fps = 1 / dt if dt > 0 else 0

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("UR5e GoPro Vision", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                filename = f"capture_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        try:
            camera_command("stop")
        except requests.RequestException as error:
            print(f"Could not stop stream: {error}")


if __name__ == "__main__":
    main()
