import numpy as np
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
import cv2
from skimage.morphology import medial_axis


def inpaint(img):
    binary_mask_scaled = img * 255
    inpaint_mask = cv2.bitwise_not(binary_mask_scaled)
    kernel = np.ones((3, 3), np.uint8)
    inpaint_mask_dilated = cv2.dilate(inpaint_mask, kernel, iterations=1)
    inpainted_image = cv2.inpaint(binary_mask_scaled, inpaint_mask_dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_image

from copy import deepcopy as DC
def watershed(img):

    watershed_mask = DC(img*255)
    dist_transform = cv2.distanceTransform(watershed_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(watershed_mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(watershed_mask * 255, cv2.COLOR_GRAY2BGR), markers)
    watershed_mask[markers == -1] = 0
    return watershed_mask


def close_broken_vessels(segmented_vessels, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_vessels = cv2.morphologyEx(segmented_vessels, cv2.MORPH_CLOSE, kernel)
    return closed_vessels

