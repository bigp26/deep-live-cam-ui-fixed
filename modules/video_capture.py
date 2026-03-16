import cv2


class VideoCapturer:
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.cap = None

    def start(self, width=640, height=480, fps=30):

        # Open the real Linux camera device directly
        self.cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open /dev/video0")

        # Set resolution and framerate
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        return True

    def read(self):
        if not self.cap:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
