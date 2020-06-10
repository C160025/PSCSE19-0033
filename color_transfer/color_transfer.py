# OpenCv package
import cv2
from color_transfer.models import Mean_CX, MKL_CX, IDT_CX, REGRAIN_CX

def ColorXfer(source, target, model, conversion=None):
    """
    Color transfer from target image's color characteristics into source image,
    by the selection color space conversion model.
    :param source: source image in RGB color space (0-255) on numpy array
    :param target: target image in RGB color space (0-255) on numpy array
    :param conversion: two type color space conversions
                  'opencv' = opencv-python package
                  'matrix' = equation referencing from Color Transfer between Images by Erik Reinhard's paper
                             http://erikreinhard.com/papers/colourtransfer.pdf
    :param model: two type conversion models
                  'mean' = compute using mean and standard deviation referencing from
                           Color Transfer between Images by Erik Reinhard's paper
                           http://erikreinhard.com/papers/colourtransfer.pdf
                  'pdf' = compute using
                  https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
                  'idt' = compute using
                  Automated Colour Grading using Colour Distribution Transfer by F. Pitié
                  https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
                  'mkl' = compute using
                  The Linear Monge-Kantorovitch Linear Colour Mapping for Example-Based Colour Transfer by F. Pitié
                  https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cvmp.pdf
    :return: output_rgb: corrected image in RGB color space (0-255) on numpy array
    """

    if model == 'mean':
        return Mean_CX(source, target, conversion)
    if model == 'mkl':
        return MKL_CX(source, target, conversion)
    if model == 'idt':
        return IDT_CX(source, target, conversion)
    if model == 'regrain':
        return REGRAIN_CX(source, target)



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
# ColorXfer
