import cv2
from color_transfer.utils import cx_rgb2lab, cx_lab2rgb


def OpenCV_CX(source, target):
    image_opencv_bgr = cv2.imread(source, cv2.IMREAD_COLOR)
    image_opencv_rgb = cv2.cvtColor(image_opencv_bgr, cv2.COLOR_BGR2RGB)
    return image_opencv_rgb

def Matrix_CX(source, target):
    test13 = cx_rgb2lab(source, True)
    height = test13.shape[0]
    width = test13.shape[1]
    test13.reshape()
    print(test13[:1])
    test14 = cx_rgb2lab(test13, True)

    return 0