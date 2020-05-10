"""
instantiate the class with some model and all the necessary function argument,
example like shape or size or background colour,
"""

import os
import math
import numpy as np
# OpenCv package
import cv2
# Scikit-image
from scipy import misc
# Pillow package
from PIL import Image
import matplotlib.pyplot as plt


# from __future__ import print_function
# from keras_vggface.models import RESNET50, VGG16, SENET50


def color_space_conversion(array, input_to_output, conversion_type):
    """
    Conversion color space from input array into output array,
     with input_to_output and conversion_type parameter selection.

    :param array: input array in Numpy array
    :param input_to_output: "rgb2lab", "lab2rgb"
    :param conversion_type: "opencv", "matrix"
    :return:
    """
    output = 0
    if input_to_output == "rgb2lab":
        if conversion_type == "opencv":
            output = cv2.cvtColor(array, cv2.COLOR_BGR2LAB).astype("float32")
        elif conversion_type == "matrix":
            output = rgb2lab(array)
    elif input_to_output == "lab2rgb":
        if conversion_type == "opencv":
            output = cv2.cvtColor(array.astype("uint8"), cv2.COLOR_LAB2BGR)
        elif conversion_type == "matrix":
            output = lab2rgb(array)

    return output

def rgb2lab(array):
    """

    :param array: input array in Numpy array
    :return:
    """
    # num = 0
    # RGB = [0, 0, 0]
    #
    # for value in array:
    #     RGB[num] = value
    #     num = num + 1
    #
    # LMS = [0, 0, 0]
    #
    # L = RGB[0] * 0.3811 + RGB[1] * 0.5783 + RGB[2] * 0.0402
    # M = RGB[0] * 0.1967 + RGB[1] * 0.7244 + RGB[2] * 0.0782
    # S = RGB[0] * 0.0241 + RGB[1] * 0.1288 + RGB[2] * 0.8444
    #
    # LMS[0] = math.log10(L)
    # LMS[1] = math.log10(M)
    # LMS[2] = math.log10(S)
    #
    # A = [[12, 7, 3],
    #      [4, 5, 6],
    #      [7, 8, 9]]
    #
    # XYZ[0] = float(XYZ[0]) / 95.047  # ref_X =  95.047   Observer= 2°, Illuminant= D65
    # XYZ[1] = float(XYZ[1]) / 100.0  # ref_Y = 100.000
    # XYZ[2] = float(XYZ[2]) / 108.883  # ref_Z = 108.883
    #
    # num = 0
    # for value in XYZ:
    #
    #     if value > 0.008856:
    #         value = value ** (0.3333333333333333)
    #     else:
    #         value = (7.787 * value) + (16 / 116)
    #
    #     XYZ[num] = value
    #     num = num + 1
    #
    # Lab = [0, 0, 0]
    #
    # L = (116 * XYZ[1]) - 16
    # a = 500 * (XYZ[0] - XYZ[1])
    # b = 200 * (XYZ[1] - XYZ[2])
    #
    # Lab[0] = round(L, 4)
    # Lab[1] = round(a, 4)
    # Lab[2] = round(b, 4)

    output = 0
    return output


def lab2rgb(array):
    """

    :param array: input array in Numpy array
    :return:
    """
    output = 0
    return output

sour = "../images/autumn.jpg"
# tar = "../images/autumnT.jpg"
#
im_pillow = np.array(Image.open(sour)) #RGB
#print(im_pillow)
# print(image.size)
# print(image.mode)

im_bgr = cv2.cvtColor(im_pillow, cv2.COLOR_RGB2BGR)
# print(im_bgr)
data = cv2.imread(sour)   #BGR
# print(data)
# print(data.format)
# print(data.size)
# print(data.mode)
#from skimage.color import rgb2lab, lab2rgb
#lab = rgb2lab(im_pillow)
#print(lab)
# img_s = max(img_s,1/255);

opencv = cv2.cvtColor(data, cv2.COLOR_BGR2LAB)#.astype("float32") #BGR to LAB
print(opencv)
num = 0
RGB = [1, 1, 1]
lab = [0, 0, 0]

a = [[0.3811, 0.5783, 0.0402],
     [0.1967, 0.7244, 0.0782],
     [0.0241, 0.1288, 0.8444]]
b = [[1/math.sqrt(3), 0, 0],
     [0, 1/math.sqrt(6), 0],
     [0, 0, 1/math.sqrt(2)]]
c = [[1, 1, 1],
     [1, 1, -2],
     [1, -1, 0]]
# print(np.dot(a, RGB))
# test = []
# for row in image:
#     test.append([])
#     for rgb in row:
#         print("rgb")
#         print(rgb)
#         lms = np.dot(a, rgb)
#         print("lms")
#         print(lms)
#         lms_b = np.around(np.log10(lms), decimals=4)
#         print("lms_b")
#         print(lms_b)
#         #np.dot(np.dot(b, c), lms_b)
#         print("lab")
#         print(np.around(np.dot(np.dot(b, c), lms_b), decimals=4))
#         #lms_ = np.append(np.log10(lms[0]),  np.log10(lms[1]),  np.log10(lms[3]))
#         # print(np.log10(lms))
#        # lms_ = np.log10(lms) if lms != 0 else lms
#        # test.append(np.dot(np.dot(b, c), lms_))


# print(test)
#     lms = np.dot(a, value)
#     lms_ = np.log10(lms)
#     lab[num] = np.dot(np.dot(b, c), lms_)
#     num = num + 1
#
# print(lab)
# # test = color_space_conversion(data, input_to_output="rgb2lab", conversion_type="opencv")
# # print(data)
# RGB = [1, 2, 3]

# lms = np.dot(a, RGB)
# lms_ = np.log10(lms)
# lab = np.dot(np.dot(b, c), lms_)
# print(lab)


# print(type(data))
# print(data.shape)
#
# image2 = Image.fromarray(data)
# print('****')
# print(type(image2))
#
# # summarize image details
# print(image2.mode)
# print(image2.size)
# show the image


def color_transfer(source, target, clip=True, preserve_paper=True):
    """
    Color transfer the image's colour characteristics from target into source.
    Converting color space with D65 matrix

    :param source: source image in Numpy array
    :param target: target image in Numpy array
    :param preserve_paper: scale by the standard deviations in boolean datatype
    :param clip: scale down the color brightness in boolean datatype
    :return:
    """
    dirname = os.path.dirname(source)
    img = Image.open(dirname)
    result = np.array(img)
    return result[20, 30]


def color_transfer_scikit(source, target, clip=True, preserve_paper=True):
    """
    Color transfer the image's colour characteristics from target into source.
    Converting color space with Scikit-image package.

    :param source: source image in Numpy array datatype
    :param target: target image in Numpy array datatype
    :param preserve_paper: scale by the standard deviations in boolean datatype
    :param clip: scale down the color brightness in boolean datatype
    :return:
    """

    result = 0
    return result


def color_transfer_orginal(source, target, clip=True, preserve_paper=True):
    """
    Color transfer the image's colour characteristics from target into source.
    Converting color space with OpenCV package.

    :param source: source image in Numpy array datatype
    :param target: target image in Numpy array datatype
    :param preserve_paper: scale by the standard deviations in boolean datatype
            (statistics and color correction section on the paper)
    :param clip: scale down the color brightness in boolean datatype

    :return:

    clip: Should components of L*a*b* image be scaled by np.clip before
    	converting back to BGR color space?
    	If False then components will be min-max scaled appropriately.
    	Clipping will keep target image brightness truer to the input.
    	Scaling will adjust image brightness to avoid washed out portions
    	in the resulting color transfer that can be caused by clipping.
    preserve_paper: Should color transfer strictly follow methodology
    	layed out in original paper? The method does not always produce
    	aesthetically pleasing results.
    	If False then L*a*b* components will scaled using the reciprocal of
    	the scaling factor proposed in the paper.  This method seems to produce
    	more consistently aesthetically pleasing results
    Returns:
    transfer = NumPy array
    	OpenCV image (w, h, 3) NumPy array (uint8)
    """
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    if preserve_paper:
        # scale by the standard deviations using paper proposed factor
        l = (lStdTar / lStdSrc) * l
        a = (aStdTar / aStdSrc) * a
        b = (bStdTar / bStdSrc) * b
    else:
        # scale by the standard deviations using reciprocal of paper proposed factor
        l = (lStdSrc / lStdTar) * l
        a = (aStdSrc / aStdTar) * a
        b = (bStdSrc / bStdTar) * b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip/scale the pixel intensities to [0, 255] if they fall outside this range
    l = _scale_array(l, clip=clip)
    a = _scale_array(a, clip=clip)
    b = _scale_array(b, clip=clip)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return transfer


def image_stats(image):
    """
	Parameters:
	-------
	image: NumPy array
		OpenCV image in L*a*b* color space
	Returns:
	-------
	Tuple of mean and standard deviations for the L*, a*, and b*
	channels, respectively
	"""
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def _min_max_scale(arr, new_range=(0, 255)):
    """
	Perform min-max scaling to a NumPy array
	Parameters:
	-------
	arr: NumPy array to be scaled to [new_min, new_max] range
	new_range: tuple of form (min, max) specifying range of
		transformed array
	Returns:
	-------
	NumPy array that has been scaled to be in
	[new_range[0], new_range[1]] range
	"""
    # get array's current min and max
    mn = arr.min()
    mx = arr.max()

    # check if scaling needs to be done to be in new_range
    if mn < new_range[0] or mx > new_range[1]:
        # perform min-max scaling
        result = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
    else:
        # return array if already in range
        result = arr

    return result


def _scale_array(arr, clip=True):
    """
	Trim NumPy array values to be in [0, 255] range with option of
	clipping or scaling.
	Parameters:
	-------
	arr: array to be trimmed to [0, 255] range
	clip: should array be scaled by np.clip? if False then input
		array will be min-max scaled to range
		[max([arr.min(), 0]), min([arr.max(), 255])]
	Returns:
	-------
	NumPy array that has been scaled to be in [0, 255] range
	"""
    if clip:
        scaled = np.clip(arr, 0, 255)
    else:
        scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
        scaled = _min_max_scale(arr, new_range=scale_range)

    return scaled
