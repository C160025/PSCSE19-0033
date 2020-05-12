import os
import math
import numpy as np
# OpenCv package
import cv2
# Scikit-image
from scipy import misc
from skimage import color, io
# Pillow package
from PIL import Image, ImageCms
import matplotlib.pyplot as plt
import time
source = "../images/autumn.jpg" # fallingwater.
source1 = "../images/fallingwater.jpg" # fallingwater.
target = "../images/autumnT.jpg"


# Scikit-image read in RGB return array tried converting RGB [45, 1, 0] to lab [5.23712195 21.32443839  8.25209207]
# match online converter result but reverse converting lab [5.23712195 21.32443839  8.25209207] to
# RGB get [1.76470588e-01 3.92156863e-03 2.32662719e-17] failed result
# tried on mathlab get the same result ????
# tried http://colormine.org/color-converter CIE-L*ab back to RGB it work

# Scikit-image read in RGB return array
image_skimage_rgb = io.imread(source)
# print(f'{image_skimage_rgb.min()}, {image_skimage_rgb.max()}')
# height width, channel
# print('Scikit-shape', image_skimage.shape)
# print('Scikit-dtype', image_skimage.dtype)
# print('Scikit-array')
# print(image_skimage[:3])
# array_image_skimage = image_skimage
# 'D65', '2' parameter
green = np.int32([[[0, 254, 0]]])
# print(green)
# print(f'{green.min()}, {green.max()}')
image_skimage_lab = color.rgb2lab(green)
# print(image_skimage_lab)
# print(f'{image_skimage_lab.min()}, {image_skimage_lab.max()}')
# print(l)
# print(f'{l.min()}, {l.max()}')
# print("image_skimage_lab", image_skimage_lab[0])
# image_skimage_lch = color.lab2lch(image_skimage_lab)
# print("image_skimage_lch", image_skimage_lch[0])
# image_skimage_lab2 = color.lch2lab(image_skimage_lch)
# print("image_skimage_lab2", image_skimage_lab2[0])
image_skimage_rgb2 = color.lab2rgb(image_skimage_lab.astype("uint8"))
# print("image_skimage_rgb2", image_skimage_rgb2[0])
# lab_green = np.array([[[87.73509949, -86.18302974,  83.17970318]]])
# output = cv2.cvtColor(array, cv2.COLOR_BGR2LAB).astype("float32")
# output = cv2.cvtColor(array.astype("uint8"), cv2.COLOR_LAB2BGR)
# print(test2)
# skimage_lab = color.convert_colorspace(array_image_skimage, 'rgb', 'rgb cie')
# print(skimage_lab[:1])

# print(skimage_lab)
# test = color.lab2rgb(skimage_lab, 'D65')
# print(test[:1])


# OpenCV read in BGR return array
image_opencv_bgr = cv2.imread(source, cv2.IMREAD_COLOR)
opencv_bgr = image_opencv_bgr

# print(image_opencv_bgr[:, :, 0])
# print(image_opencv_bgr[:, :, 1])
# print(image_opencv_bgr[:, :, 2])
image_opencv_xyz = cv2.cvtColor(image_opencv_bgr, cv2.COLOR_RGB2XYZ)
# print(f'{image_opencv_xyz.min()}, {image_opencv_xyz.max()}')
image_opencv_lab = cv2.cvtColor(image_opencv_bgr, cv2.COLOR_BGR2Lab)
# print(image_opencv_lab[:1])
# print(f'{image_opencv_lab.min()}, {image_opencv_lab.max()}')
# image_opencv_bgr2 = cv2.cvtColor(image_opencv_lab.astype("int32"), cv2.COLOR_LAB2BGR)
# print(image_opencv_bgr2[:1])
# print(f'{image_opencv_bgr2.min()}, {image_opencv_bgr2.max()}')
# print('OpenCV-size', image_opencv.shape)
# print('OpenCV-dtype', image_opencv.dtype)
# correct BGR to RGB
# challenges and difficulties need additional conversion from BGR to RGB
# image_opencv_rgb = cv2.cvtColor(image_opencv_bgr, cv2.COLOR_BGR2RGB)
# print(image_opencv_rgb[:1])
# image_opencv_lab = cv2.cvtColor(image_opencv_rgb, cv2.COLOR_RGB2Lab).astype("float32")
# print(image_opencv_lab[:1])
# image_opencv_bgr2 = cv2.cvtColor(image_opencv_lab.astype("uint8"), cv2.COLOR_LAB2RGB)
# print(image_opencv_bgr2[:1])
#plt.imshow(array_image_skimage)
#plt.show()
# print('OpenCV-array')
# print(image_opencv_RGB[:3])
# print(array_image_opencv_BGR[:1])
# plt.imshow(array_image_opencv_BGR)
# plt.show()

# print(openvs_lab_BGR[:1])
# print(test3[:1])
# print(test2[:1])
# plt.imshow(test2)
# plt.show()

# print(opencv_RGB2lab[:1])


# Pillow read in RGB return object is very bad for multiplication
# when is an object and lack of converting colour space function
# can put into project report challenges and difficulties
image_pillow = Image.open(source).convert('RGB')
# print('Pillow-size', image_pillow.size)
# print('Pillow-array')
# print(np.uint8(image_pillow)[:1])
# print(np.uint8(Lab)[:1])
# L, a, b = Lab.split()

array_image_pillow = np.array(image_pillow)[:1]






image_opencv_bgr = cv2.imread(source, cv2.IMREAD_COLOR)
image_opencv_rgb = cv2.cvtColor(image_opencv_bgr, cv2.COLOR_BGR2RGB)
test13 = ct_rgb2lab(image_opencv_rgb, True)
height = test13.shape[0]
width = test13.shape[1]
test13.reshape()
print(test13[:1])
test14 = ct_lab2rgb(test13, True)
test15 = np.rint(test14)
test15[test15 <= -0.] = 0.
print(test15[:1])
# test = np.array([[[44.99915824, 0.99912331, 0.0069295]]], dtype=np.float32)
# test12 = np.rint(test)
# print(test12)

# print(image.size)
# print(image.mode)
#
# im_bgr = cv2.cvtColor(im_pillow, cv2.COLOR_RGB2BGR)
# # print(im_bgr)
# data = cv2.imread(source)   #BGR
# # print(data)
# # print(data.format)
# # print(data.size)
# # print(data.mode)
# #lab = rgb2lab(im_pillow)
# #print(lab)
# # img_s = max(img_s,1/255);
#
# opencv = cv2.cvtColor(data, cv2.COLOR_BGR2LAB)#.astype("float32") #BGR to LAB
# print(opencv)
# num = 0
# RGB = [1, 1, 1]
# lab = [0, 0, 0]
#
# a = [[0.3811, 0.5783, 0.0402],
#      [0.1967, 0.7244, 0.0782],
#      [0.0241, 0.1288, 0.8444]]
# b = [[1/math.sqrt(3), 0, 0],
#      [0, 1/math.sqrt(6), 0],
#      [0, 0, 1/math.sqrt(2)]]
# c = [[1, 1, 1],
#      [1, 1, -2],
#      [1, -1, 0]]
# # print(np.dot(a, RGB))
# # test = []
# # for row in image:
# #     test.append([])
# #     for rgb in row:
# #         print("rgb")
# #         print(rgb)
# #         lms = np.dot(a, rgb)
# #         print("lms")
# #         print(lms)
# #         lms_b = np.around(np.log10(lms), decimals=4)
# #         print("lms_b")
# #         print(lms_b)
# #         #np.dot(np.dot(b, c), lms_b)
# #         print("lab")
# #         print(np.around(np.dot(np.dot(b, c), lms_b), decimals=4))
# #         #lms_ = np.append(np.log10(lms[0]),  np.log10(lms[1]),  np.log10(lms[3]))
# #         # print(np.log10(lms))
# #        # lms_ = np.log10(lms) if lms != 0 else lms
# #        # test.append(np.dot(np.dot(b, c), lms_))
#
#
# # print(test)
# #     lms = np.dot(a, value)
# #     lms_ = np.log10(lms)
# #     lab[num] = np.dot(np.dot(b, c), lms_)
# #     num = num + 1
# #
# # print(lab)
# # # test = color_space_conversion(data, input_to_output="rgb2lab", conversion_type="opencv")
# # # print(data)
# # RGB = [1, 2, 3]
#
# # lms = np.dot(a, RGB)
# # lms_ = np.log10(lms)
# # lab = np.dot(np.dot(b, c), lms_)
# # print(lab)
#
#
# # print(type(data))
# # print(data.shape)
# #
# # image2 = Image.fromarray(data)
# # print('****')
# # print(type(image2))
# #
# # # summarize image details
# # print(image2.mode)
# # print(image2.size)
# # show the image
