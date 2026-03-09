
import cv2
import numpy as np

class FaceCropper:
    def __init__(self, target_size=256):
        self.target_size = target_size

    def crop_center_square(self, image):
        h, w = image.shape[:2]
        size = min(h, w, self.target_size)  # never upsample

        x1 = (w - size) // 2
        y1 = (h - size) // 2

        return image[y1:y1 + size, x1:x1 + size]

