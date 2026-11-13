# Handles YOLOv8 camera feed & object detection

import cv2
import numpy as np

class VisionProcess:

    # This is to initiliaze the container for the CV logic by making the screen
    # same as my 16:9 inch dashboard display size for YOLOv8
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        print("Processer has  initialized properlyys!")

    def process_carla_image(self, image):

        # This converts the memory from CARLA into a flat numpy array of numbers so that the CV can read it
        # also reshape the image to match the same type as the dashboard obv
        image_array = np.array(image.raw_data)
        image_reshape =  image_array.reshape((self.height,self.width,4))

        # This slice keeps the row and columns but only the first 3 (Blue, green, yellow) for video feeds
        bgr_frame = image_reshape[: , : , :3].copy()

        return bgr_frame

    def apply_cv_pipline(self, frame):
        # FUTURE YOLOv8 aand PARKING TYPE DETECTOR CODE SECTION HERE FOR FINAL PART
        # Future section here must include being able to detect obstacles (what kind) and the slots and determine which parking to take

        # test draw a text on the camera feed
        cv2.putText(
            frame, "SpatialPark : rear-view working", (50,50), cv2.FONT_HERSHEY_SIMPLEX,
            1.0,(0,255,0),2
        )

        return frame