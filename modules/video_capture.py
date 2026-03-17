import os
import platform

import cv2


class VideoCapturer:
    def __init__(self, device_index: int = 0):
        self.device_index = int(device_index) if device_index is not None else 0
        self.cap: cv2.VideoCapture | None = None

    def start(self, width: int = 640, height: int = 480, fps: int = 30) -> bool:
        """
        Open camera in a platform-correct way.

        Linux:
            - first try integer device index with V4L2
            - then fallback to /dev/video{index} if needed

        Other platforms:
            - use integer device index directly
        """
        self.release()

        sysname = platform.system().lower()
        opened = False

        if sysname == "linux":
            # Preferred path: respect the selected device index.
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
            opened = bool(self.cap is not None and self.cap.isOpened())

            # Fallback path: explicit V4L2 device file for the chosen index.
            if not opened:
                dev_path = f"/dev/video{self.device_index}"
                if os.path.exists(dev_path):
                    self.cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
                    opened = bool(self.cap is not None and self.cap.isOpened())
        else:
            self.cap = cv2.VideoCapture(self.device_index)
            opened = bool(self.cap is not None and self.cap.isOpened())

        if not opened or self.cap is None:
            raise RuntimeError(
                f"Failed to open camera device index={self.device_index}"
            )

        # Low-latency hint where supported.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))

        return True

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None
