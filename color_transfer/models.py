import cv2
import numpy as np
from color_transfer.utils import cx_rgb2lab, cx_lab2rgb

def OpenCV_CX(source_bgr, target_bgr):
    """
    Color transfer from target image's color characteristics into source image,
    using opencv-python package to convert between color space.
    :param source_bgr: source image in BGR color space on numpy array
    :param target_bgr: target image in BGR color space on numpy array
    :return: output_bgr: corrected image in BGR color space on numpy array
    """
    # convert from BGR to LAB color space for source and target
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype("float32")

    # statistics and color correction
    correction = color_correction(source_lab, target_lab)

    # convert from LAB to BGR color space
    output_bgr = cv2.cvtColor(correction.astype("uint8"), cv2.COLOR_LAB2BGR)

    return output_bgr

def Matrix_CX(source_rgb, target_rgb):
    """
    Color transfer from target image's color characteristics into source image,
    Referencing from https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf paper
    :param source_rgb: source image in RGB color space on numpy array
    :param target_rgb: target image in RGB color space on numpy array
    :return: output_bgr: corrected image in BGR color space on numpy array
    """
    # convert from RGB to LAB color space for source and target
    source_lab = cx_rgb2lab(source_rgb, True)
    target_lab = cx_rgb2lab(target_rgb, True)

    # statistics and color correction
    correction = color_correction(source_lab, target_lab, True)

    # convert from LAB to RGB color space
    output_rgb = cx_lab2rgb(correction, True)
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

    return output_bgr

def color_correction(source_lab, target_lab, clip='False'):
    """
    Color correction is to compute mean and standard deviation for each axis
    individually in the lab color space.
    Referencing from https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf paper
    :param source_lab: source image in lab color space on numpy array
    :param target_lab: target image in lab color space on numpy array
    :return: output_lab: corrected image in lab color space on numpy array
    """
    # split into individual channel space for statistics and color correction
    (source_l, source_a, source_b) = cv2.split(source_lab)
    (target_l, target_a, target_b) = cv2.split(target_lab)

    # equation 10 subtract source mean from the source data points
    l = source_l - source_l.mean()
    a = source_a - source_a.mean()
    b = source_b - source_b.mean()

    # equation 11 scale down by factoring the respective standard deviation
    l = (target_l.std() / source_l.std()) * l
    a = (target_a.std() / source_a.std()) * a
    b = (target_b.std() / source_b.std()) * b

    # adding the target mean back to the respective data points
    l = l + target_l.mean()
    a = a + target_a.mean()
    b = b + target_b.mean()

    # limit the data points range for opencv-python conversion lab to BGR
    if(clip):
        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)

    # merge individual channel back into lab color space
    output_lab = cv2.merge([l, a, b])

    return output_lab


