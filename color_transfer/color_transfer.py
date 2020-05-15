# OpenCv package
import cv2
from color_transfer.models import OpenCV_CX, Matrix_CX

def ColorXfer(source, target, model):
    """
    Color transfer from target image's color characteristics into source image,
    by the selection color space conversion model.
    :param source: path and name of source image
    :param target: path and name of target image
    :param model: two type conversion models
                  'opencv' = opencv-python package
                  'matrix' = equation referencing from https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf paper
    :return: output image in RGB color space
    """
    source_bgr = cv2.imread(source, cv2.IMREAD_COLOR)
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_RGB2BGR)
    target_bgr = cv2.imread(target, cv2.IMREAD_COLOR)
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_RGB2BGR)

    if model == 'opencv':
        return OpenCV_CX(source_bgr, target_bgr)
    if model == 'matrix':
        return Matrix_CX(source_rgb, target_rgb)




# im_pillow = np.array(Image.open(sour)) # RGB
# print(im_pillow)
# print(image.size)
# print(image.mode)

# im_bgr = cv2.cvtColor(im_pillow, cv2.COLOR_RGB2BGR)
# print(im_bgr)
# data = cv2.imread(sour)   # BGR
# print(data)
# print(data.format)
# print(data.size)
# print(data.mode)
# from skimage.color import rgb2lab, lab2rgb
# lab = rgb2lab(im_pillow)
# print(lab)
# img_s = max(img_s,1/255);
