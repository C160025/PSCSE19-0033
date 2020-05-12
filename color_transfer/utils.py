
import os
import math
import numpy as np
import time
# OpenCv package
import cv2
# Scikit-image
from scipy import misc
# Pillow package
from PIL import Image



def cx_rgb2lab(image_rgb, log10):
    """

    :param image_rgb:
    :param log10:
    :return:
    """
    height = image_rgb.shape[0]
    width = image_rgb.shape[1]
    img_lab = np.zeros_like(image_rgb, dtype=np.float32)
    # equation 2 to convert RGB space to XYZ space
    rgb2xyz_eq = [[0.5141, 0.3239, 0.1604],
                  [0.2651, 0.6702, 0.0641],
                  [0.0241, 0.1228, 0.8444]]
    # equation 3 to convert XYZ space to LMS space
    xyz2lms_eq = [[0.3897, 0.6890, -0.0787],
                  [-0.2298, 1.1834, 0.0464],
                  [0.0000, 0.0000, 1.0000]]
    # dot product of equation 2 and 3 to convert RGB space to LMS space (more precise)
    rgb2lms_eq1 = np.matmul(xyz2lms_eq, rgb2xyz_eq)
    # equation 4 to to convert RGB space to LMS space
    rgb2lms_eq2 = [[0.3811, 0.5783, 0.0402],
                   [0.1967, 0.7244, 0.0782],
                   [0.0241, 0.1288, 0.8444]]
    # left of equation 6 to convert LMS space to lab space
    lms2lab_eq1 = [[1/math.sqrt(3), 0.0000, 0.0000],
                   [0.0000, 1/math.sqrt(6), 0.0000],
                   [0.0000, 0.0000, 1/math.sqrt(2)]]
    # right of equation 6 to convert LMS space to lab space
    lms2lab_eq2 = [[1.0000, 1.0000, 1.0000],
                   [1.0000, 1.0000, -2.0000],
                   [1.0000, -1.0000, 0.0000]]
    # dot product of equation 6 to convert LMS space to lab space
    lms2lab_eq = np.matmul(lms2lab_eq1, lms2lab_eq2)
    t = time.time()
    for x in range(0, width):
        for y in range(0, height):
            # convert RGB space to LMS space
            img_lab[y, x] = np.matmul(rgb2lms_eq1, image_rgb[y, x])
            if (log10):
                # equation 5 to eliminate the skew
                img_lab[y, x] = np.log10(img_lab[y, x])
            # convert LMS space to lab space
            img_lab[y, x] = np.matmul(lms2lab_eq, img_lab[y, x])

    print("took {} s".format(time.time() - t))
    return img_lab




def cx_lab2rgb(image_lab, power10):
    """

    :param image_lab: numpy array
    :param power10:
    :return:
    """
    height = image_lab.shape[0]
    width = image_lab.shape[1]
    img_rgb = np.zeros_like(image_lab, dtype=np.float32)
    # left of equation 8 to convert lab space to LMS space
    lab2lms_eq1 = [[1.0000, 1.0000, 1.0000],
                   [1.0000, 1.0000, -1.0000],
                   [1.0000, -2.0000, 0.0000]]
    # right of equation 8 to convert lab space to LMS space
    lab2lms_eq2 = [[math.sqrt(3)/3, 0.0000, 0.0000],
                   [0.0000, math.sqrt(6)/6, 0.0000],
                   [0.0000, 0.0000, math.sqrt(2)/2]]
    # dot product of equation 8 to convert lab space to LMS space
    lab2lms_eq = np.matmul(lab2lms_eq1, lab2lms_eq2)
    # equation 9 to convert LMS space to RGB space
    lms2rgb_eq = [[4.4679, -3.5876, 0.1193],
                  [-1.2186, 2.3809, -0.1624],
                  [0.0497, -0.2439, 1.2045]]
    t = time.time()
    for x in range(0, width):
        for y in range(0, height):
            # convert lab space to LMS space
            img_rgb[y, x] = np.matmul(lab2lms_eq, image_lab[y, x])
            if (power10):
                # raise back to linear space
                img_rgb[y, x] = np.power(10, img_rgb[y, x])
            # convert LMS space to RGB space
            img_rgb[y, x] = np.matmul(lms2rgb_eq, img_rgb[y, x])
    print("took {} s".format(time.time() - t))
    return img_rgb