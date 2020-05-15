import math
import numpy as np
import cv2
import time

def cx_rgb2lab(image_rgb, log10):
    """
    Color space conversion from RGB to lab space referencing from paper
    Referencing from https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf paper
    :param image_rgb: image in RGB color space on numpy array
    :param log10:
    :return: image in lab color space on numpy array
    """
    t = time.time()
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
    lms2lab_eq1 = [[1 / math.sqrt(3), 0.0000, 0.0000],
                   [0.0000, 1 / math.sqrt(6), 0.0000],
                   [0.0000, 0.0000, 1 / math.sqrt(2)]]

    # right of equation 6 to convert LMS space to lab space
    lms2lab_eq2 = [[1.0000, 1.0000, 1.0000],
                   [1.0000, 1.0000, -2.0000],
                   [1.0000, -1.0000, 0.0000]]

    # dot product of equation 6 to convert LMS space to lab space
    lms2lab_eq = np.matmul(lms2lab_eq1, lms2lab_eq2)

    # split RGB into individual channel space for color space conversion
    (r, g, b) = cv2.split(image_rgb)

    # convert RGB space to LMS space
    L = r * rgb2lms_eq1[0][0] + g * rgb2lms_eq1[0][1] + b * rgb2lms_eq1[0][2]
    M = r * rgb2lms_eq1[1][0] + g * rgb2lms_eq1[1][1] + b * rgb2lms_eq1[1][2]
    S = r * rgb2lms_eq1[2][0] + g * rgb2lms_eq1[2][1] + b * rgb2lms_eq1[2][2]

    # equation 5 to eliminate the skew
    if (log10):
        L = np.log10(L)
        M = np.log10(M)
        S = np.log10(S)

    # convert LMS space to lab space
    l = L * lms2lab_eq[0][0] + M * lms2lab_eq[0][1] + S * lms2lab_eq[0][2]
    a = L * lms2lab_eq[1][0] + M * lms2lab_eq[1][1] + S * lms2lab_eq[1][2]
    b = L * lms2lab_eq[2][0] + M * lms2lab_eq[2][1] + S * lms2lab_eq[2][2]

    # merge individual channel into lab color space
    image_lab = cv2.merge([l, a, b]).astype(np.float32)

    print("took {} s".format(time.time() - t))
    return image_lab

def cx_lab2rgb(image_lab, power10):
    """
    Color space conversion from lab to RGB space.
    Referencing from https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf paper
    :param image_lab: image in lab color space on numpy array
    :param power10:
    :return: image in RGB color space on numpy array
    """
    t = time.time()
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

    # split lab into individual channel space for color space conversion
    (l, a, b) = cv2.split(image_lab)

    # convert lab space to LMS space
    L = l * lab2lms_eq[0][0] + a * lab2lms_eq[0][1] + b * lab2lms_eq[0][2]
    M = l * lab2lms_eq[1][0] + a * lab2lms_eq[1][1] + b * lab2lms_eq[1][2]
    S = l * lab2lms_eq[2][0] + a * lab2lms_eq[2][1] + b * lab2lms_eq[2][2]

    # raise back to linear space
    if (power10):
        L = np.power(10, L)
        M = np.power(10, M)
        S = np.power(10, S)

    # convert LMS space to RGB space
    r = L * lms2rgb_eq[0][0] + M * lms2rgb_eq[0][1] + S * lms2rgb_eq[0][2]
    g = L * lms2rgb_eq[1][0] + M * lms2rgb_eq[1][1] + S * lms2rgb_eq[1][2]
    b = M * lms2rgb_eq[2][0] + M * lms2rgb_eq[2][1] + S * lms2rgb_eq[2][2]

    # merge individual channel into RGB color space
    img_rgb = cv2.merge([r, g, b]).astype(np.float32)

    print("took {} s".format(time.time() - t))
    return img_rgb