
import socket
import time

ROBOT_IP = "172.16.34.2"
GRIPPER_PORT = 63352


def command(sock, message):
    sock.sendall((message + "\n").encode())
    return sock.recv(1024).decode().strip()


class Gripper:
    def __init__(self):
        self.sock = None

    def connect(self):
        self.sock = socket.create_connection(
            (ROBOT_IP, GRIPPER_PORT),
            timeout=3
        )
        self.sock.settimeout(3)

        status = command(self.sock, "GET STA")

        if status != "STA 3":
            command(self.sock, "SET ACT 1")

            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                time.sleep(0.25)
                if command(self.sock, "GET STA") == "STA 3":
                    break
            else:
                raise RuntimeError(
                    "Gripper activation timed out."
                )

        command(self.sock, "SET SPE 80")
        command(self.sock, "SET FOR 40")

    def move(self, position):
        if self.sock is None:
            raise RuntimeError("Gripper not connected.")

        position = int(position)

        if not 0 <= position <= 255:
            raise ValueError("Position must be 0-255.")

        command(self.sock, f"SET POS {position}")
        command(self.sock, "SET GTO 1")

        deadline = time.monotonic() + 6

        while time.monotonic() < deadline:
            time.sleep(0.15)
            state = command(self.sock, "GET OBJ")

            if state in ("OBJ 1", "OBJ 2", "OBJ 3"):
                return state

        raise RuntimeError("Gripper motion timed out.")

    def open(self):
        return self.move(0)

    def close(self):
        return self.move(255)

    def close_connection(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def __enter__(self):
        try:
            self.connect()
            return self
        except Exception:
            self.close_connection()
            raise

    def __exit__(self, exc_type, exc, tb):
        self.close_connection()


def open_gripper():
    with Gripper() as gripper:
        return gripper.open()


def close_gripper():
    with Gripper() as gripper:
        return gripper.close()


def test():
    print("Testing Hand-E. Keep fingers and objects clear.")

    with Gripper() as gripper:
        gripper.open()
        time.sleep(1)
        gripper.close()
        print("Gripper test complete.")

