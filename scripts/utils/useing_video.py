# import pyrealsense2 as rs
import numpy as np
import cv2
import os

target = "Video"
TARGET_VIDEO = r"/home/autonav-linux/catkin_ws/src/yolov5_ROS/scripts/traffic_light_test.mp4"
TARGET_FOLDER = r"/home/autonav-linux/catkin_ws/src/yolov5_ROS/scripts/images"

# Configure depth and color streams
class Camera:

    def __init__(self):
        """
        Camera initializing part, usually insert here your camera code except loop part(l.e. get next frame)
        """
        if target == "Video":
            self.vid = cv2.VideoCapture(TARGET_VIDEO)

        elif target == "folder":
            self.img_list = os.listdir(TARGET_FOLDER)

    def get_next_img(self) -> np.ndarray:

        """
        Send Img to main code return type "must" np.ndarray
        Returns: img (np.ndarray)

        """

        # Wait for a coherent pair of frames: depth and color
        if target == "Video":
            _, frame = self.vid.read()
            return frame

        elif target == "folder":
            img = cv2.imread(fr"{TARGET_FOLDER}/{self.img_list[0]}")
            del self.img_list[0]
            return img