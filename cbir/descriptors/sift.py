import numpy as np
import cv2
from .descriptor_base import DescriptorBase


class Sift(DescriptorBase):
    def __init__(self, patch_size=65):
        super(Sift, self).__init__("data")
        self.patch_size = (int(patch_size), int(patch_size))
        self.sift = cv2.xfeatures2d.SIFT_create()

    def describe(self, img):
        kp, desc = self.sift.detectAndCompute(img, None)
        desc = np.array(desc, dtype=np.float32)
        if desc.size <= 1:
            desc = np.zeros((1, 128))
        return desc
