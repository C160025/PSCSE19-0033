import math
import numpy as np
import cv2
import time

#  F. Pitié 2007 Automated colour grading using colour distribution transfer
def rotation_matrix():
    three_dim_optimised_rotation = [
        [[1.000000, 0.000000, 0.000000], #1
         [0.000000, 1.000000, 0.000000],
         [0.000000, 0.000000, 1.000000]],
        [[0.333333, 0.666667, 0.666667], #2
         [0.666667, 0.333333, -0.666667],
         [-0.666667, 0.666667, -0.333333]],
        [[0.577350, 0.211297, 0.788682], #3
         [-0.577350, 0.788668, 0.211352],
         [0.577350, 0.577370, -0.577330]],
        [[0.577350, 0.408273, 0.707092], #4
         [-0.577350, -0.408224, 0.707121],
         [0.577350, -0.816497, 0.000029]],
        [[0.332572, 0.910758, 0.244778], #5
         [-0.910887, 0.242977, 0.333536],
         [-0.244295, 0.333890, -0.910405]],
        [[0.243799, 0.910726, 0.333376], #6
         [0.910699, -0.333174, 0.244177],
         [-0.333450, -0.244075, 0.910625]],
        [[-0.109199, 0.810241, 0.575834], #7
         [0.645399, 0.498377, -0.578862],
         [0.756000, -0.308432, 0.577351]],
        [[0.759262, 0.649435, -0.041906], #8
         [0.143443, -0.104197, 0.984158],
         [0.634780, -0.753245, -0.172269]],
        [[0.862298, 0.503331, -0.055679], #9
         [-0.490221, 0.802113, -0.341026],
         [-0.126988, 0.321361, 0.938404]],
        [[0.982488, 0.149181, 0.111631], #10
         [0.186103, -0.756525, -0.626926],
         [-0.009074, 0.636722, -0.771040]],
        [[0.687077, -0.577557, -0.440855], #11
         [0.592440, 0.796586, -0.120272],
         [-0.420643, 0.178544, -0.889484]],
        [[0.463791, 0.822404, 0.329470], #12
         [0.030607, -0.386537, 0.921766],
         [-0.885416, 0.417422, 0.204444]],
    ]




# Erik Reinhard 2001 Color Transfer between Images
def cx_rgb2lab(image_rgb, log10):
    """
    Color space conversion from RGB to lab space referencing from paper
    Referencing from http://erikreinhard.com/papers/colourtransfer.pdf paper
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

    # to prevent divide by zero encountered in log
    minval = 0.0000000001

    # equation 5 to eliminate the skew
    if (log10):
        L = np.log10(L.clip(min=minval))
        M = np.log10(M.clip(min=minval))
        S = np.log10(S.clip(min=minval))

    # convert LMS space to lab space
    l = L * lms2lab_eq[0][0] + M * lms2lab_eq[0][1] + S * lms2lab_eq[0][2]
    a = L * lms2lab_eq[1][0] + M * lms2lab_eq[1][1] + S * lms2lab_eq[1][2]
    b = L * lms2lab_eq[2][0] + M * lms2lab_eq[2][1] + S * lms2lab_eq[2][2]

    # merge individual channel into lab color space
    image_lab = cv2.merge([l, a, b]).astype(np.float32)

    print("took {} s to be remove after testing phase".format(time.time() - t))
    return image_lab

# Erik Reinhard 2001 Color Transfer between Images
def cx_lab2rgb(image_lab, power10):
    """
    Color space conversion from lab to RGB space.
    Referencing from http://erikreinhard.com/papers/colourtransfer.pdf paper
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

    print("took {} s to be remove after testing phase".format(time.time() - t))
    return img_rgb