# OpenCv package
import cv2
from color_transfer.models import Mean_CX, PDF_CX, PDF_IDT_CX, PDF_MKL_CX

def ColorXfer(source, target, model, conversion):
    """
    Color transfer from target image's color characteristics into source image,
    by the selection color space conversion model.
    :param source: path and name of source image
    :param target: path and name of target image
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
    :return: output image in RGB color space
    """
    source_bgr = cv2.imread(source, cv2.IMREAD_COLOR)
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_RGB2BGR)
    target_bgr = cv2.imread(target, cv2.IMREAD_COLOR)
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_RGB2BGR)

    if model == 'mean':
        return Mean_CX(source, target, conversion)
    if model == 'pdf':
        return PDF_CX(source_rgb, target_rgb)




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
