import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
class Camera:

    def __init__(self):
        """
        Camera initializing part, usually insert here your camera code except loop part(l.e. get next frame)
        """
        self.vid = cv2.VideoCapture("")


    def get_next_img(self) -> np.ndarray:

        """
        Send Img to main code return type "must" np.ndarray
        Returns: img (np.ndarray)

        """

        # Wait for a coherent pair of frames: depth and color
        _, frame = self.vid.read()

        return frame
