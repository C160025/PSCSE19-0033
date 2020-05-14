import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def convert_color_space_BGR_to_RGB(img_BGR):
    img_RGB = np.zeros_like(img_BGR, dtype=np.float32)
    height = img_RGB.shape[0]
    width = img_RGB.shape[1]
    tpose = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    for x in range(0, width):
        for y in range(0, height):
            img_RGB[y, x] = np.matmul(img_BGR[y, x], tpose)

    return img_RGB


def convert_color_space_RGB_to_LMS(img_RGB, log):
    conversion = [[0.5141, 0.3239, 0.1604], [0.2651, 0.6702, 0.0641], [0.0241, 0.1228, 0.8444]]
    height = img_RGB.shape[0]
    width = img_RGB.shape[1]
    img_XYZ = np.zeros_like(img_RGB, dtype=np.float32)

    img_lms = np.zeros_like(img_RGB, dtype=np.float32)
    k2 = [[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]]
    for x in range(0, width):
        for y in range(0, height):
            img_lms[y, x] = np.matmul(k2, img_RGB[y, x])
            if (log == 1):
                img_lms[y, x] = np.log10(img_lms[y, x])

    return img_lms


def convert_color_space_RGB_to_BGR(img_RGB):
    img_BGR = np.zeros_like(img_RGB, dtype=np.float32)
    height = img_BGR.shape[0]
    width = img_BGR.shape[1]
    tpose = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    for x in range(0, width):
        for y in range(0, height):
            img_BGR[y, x] = np.matmul(img_RGB[y, x], tpose)

    return img_BGR


def convert_color_space_LMS_to_RGB(img_LMS):
    k1 = [[4.4679, -3.5873, 0.1193], [-1.2186, 2.3809, -0.1624], [0.0497, -0.2439, 1.2045]]
    img_RGB = np.zeros_like(img_LMS, dtype=np.float32)
    height = img_LMS.shape[0]
    width = img_LMS.shape[1]

    for x in range(0, width):
        for y in range(0, height):
            img_RGB[y, x] = np.matmul(k1, img_LMS[y, x])

    return img_RGB


def transfer(img_src, img_tgt):
    print("\tTransfering Colors")
    hs = img_src.shape[0]
    ws = img_src.shape[1]
    ht = img_tgt.shape[0]
    wt = img_tgt.shape[1]

    TLtable = np.zeros((ht, wt), dtype=np.float32)
    TAtable = np.zeros((ht, wt), dtype=np.float32)
    TBtable = np.zeros((ht, wt), dtype=np.float32)

    SLtable = np.zeros((hs, ws), dtype=np.float32)
    SAtable = np.zeros((hs, ws), dtype=np.float32)
    SBtable = np.zeros((hs, ws), dtype=np.float32)

    img_mix = np.zeros_like(img_src, dtype=np.float32)

    for x in range(0, wt):
        for y in range(0, ht):
            TLtable[y, x] = img_tgt[y, x][0]
            TAtable[y, x] = img_tgt[y, x][1]
            TBtable[y, x] = img_tgt[y, x][2]

    for x in range(0, ws):
        for y in range(0, hs):
            SLtable[y, x] = img_src[y, x][0]
            SAtable[y, x] = img_src[y, x][1]
            SBtable[y, x] = img_src[y, x][2]

    TLmean = np.mean(TLtable)
    TAmean = np.mean(TAtable)
    TBmean = np.mean(TBtable)

    TLstd = np.std(TLtable)
    TAstd = np.std(TAtable)
    TBstd = np.std(TBtable)

    SLmean = np.mean(SLtable)
    SAmean = np.mean(SAtable)
    SBmean = np.mean(SBtable)

    SLstd = np.std(SLtable)
    SAstd = np.std(SAtable)
    SBstd = np.std(SBtable)

    for x in range(0, ws):
        for y in range(0, hs):
            img_mix[y, x][0] = (TLstd / SLstd) * (img_src[y, x][0] - SLmean) + TLmean
            img_mix[y, x][1] = (TAstd / SAstd) * (img_src[y, x][1] - SAmean) + TAmean
            img_mix[y, x][2] = (TBstd / SBstd) * (img_src[y, x][2] - SBmean) + TBmean

    return img_mix


def convert_color_space_RGB_to_Lab(img_RGB):
    '''
    convert image color space RGB to Lab
    '''
    img_LMS = convert_color_space_RGB_to_LMS(img_RGB, 1)
    img_Lab = np.zeros_like(img_RGB, dtype=np.float32)
    k1 = [[1 / np.sqrt(3), 0, 0], [0, 1 / np.sqrt(6), 0], [0, 0, 1 / np.sqrt(2)]]
    k2 = [[1, 1, 1], [1, 1, -2], [1, -1, 0]]
    height = img_LMS.shape[0]
    width = img_LMS.shape[1]

    for x in range(0, width):
        for y in range(0, height):
            img_Lab[y, x] = np.matmul(np.matmul(k1, k2), img_LMS[y, x])

    return img_Lab


def convert_color_space_Lab_to_RGB(img_Lab):
    '''
    convert image color space Lab to RGB
    '''
    k1 = [[1, 1, 1], [1, 1, -1], [1, -2, 0]]
    k2 = [[np.sqrt(3) / 3, 0, 0], [0, np.sqrt(6) / 6, 0], [0, 0, np.sqrt(2) / 2]]
    img_LMS = np.zeros_like(img_Lab, dtype=np.float32)
    height = img_Lab.shape[0]
    width = img_Lab.shape[1]

    for x in range(0, width):
        for y in range(0, height):
            img_LMS[y, x] = np.power(10, np.matmul(np.matmul(k1, k2), img_Lab[y, x]))

    img_RGB = convert_color_space_LMS_to_RGB(img_LMS)

    return img_RGB


def convert_color_space_RGB_to_CIECAM97s(img_RGB):
    '''
    convert image color space RGB to CIECAM97s
    '''
    img_RGB = convert_color_space_RGB_to_LMS(img_RGB, 0)
    img_CIECAM97s = np.zeros_like(img_RGB, dtype=np.float32)
    k1 = [[2.00, 1.00, 0.05], [1.00, -1.09, 0.09], [0.11, 0.11, -0.22]]
    height = img_RGB.shape[0]
    width = img_RGB.shape[1]

    for x in range(0, width):
        for y in range(0, height):
            img_CIECAM97s[y, x] = np.matmul(k1, img_RGB[y, x])

    return img_CIECAM97s


def convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s):
    '''
    convert image color space CIECAM97s to RGB
    '''
    img_RGB = np.zeros_like(img_CIECAM97s, dtype=np.float32)

    k1 = [[0.3279, 0.3216, 0.2061], [0.3279, -0.6343, -0.1854], [0.3279, -0.1569, -4.535]]
    height = img_RGB.shape[0]
    width = img_RGB.shape[1]

    for x in range(0, width):
        for y in range(0, height):
            img_RGB[y, x] = np.matmul(k1, img_CIECAM97s[y, x])

    img_RGB = convert_color_space_LMS_to_RGB(img_RGB)

    return img_RGB


def color_transfer_in_Lab(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')
    img_RGB_new_Lab = convert_color_space_BGR_to_RGB(img_RGB_source)
    img_RGB_new_Lab = convert_color_space_RGB_to_Lab(img_RGB_new_Lab)

    img_tget = convert_color_space_BGR_to_RGB(img_RGB_target)
    img_tget = convert_color_space_RGB_to_Lab(img_tget)

    img_RGB_new_Lab = transfer(img_RGB_new_Lab, img_tget)

    img_RGB_new_Lab = convert_color_space_Lab_to_RGB(img_RGB_new_Lab)
    img_RGB_new_Lab = convert_color_space_RGB_to_BGR(img_RGB_new_Lab)

    return img_RGB_new_Lab


def color_transfer_in_RGB(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_RGB =====')
    img_src = convert_color_space_BGR_to_RGB(img_RGB_source)
    img_tget = convert_color_space_BGR_to_RGB(img_RGB_target)

    img_src = transfer(img_src, img_tget)

    img_src = convert_color_space_RGB_to_BGR(img_src)

    return img_src


def color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_CIECAM97s =====')
    img_src = convert_color_space_BGR_to_RGB(img_RGB_source)
    img_tget = convert_color_space_BGR_to_RGB(img_RGB_target)

    img_src = convert_color_space_RGB_to_CIECAM97s(img_src)
    img_tget = convert_color_space_RGB_to_CIECAM97s(img_tget)

    img_src = transfer(img_src, img_tget)

    img_src = convert_color_space_CIECAM97s_to_RGB(img_src)
    img_src = convert_color_space_RGB_to_BGR(img_src)

    return img_src


def color_transfer(img_RGB_source, img_RGB_target, option):
    if option == 'in_RGB':
        img_RGB_new = color_transfer_in_RGB(img_RGB_source, img_RGB_target)
    elif option == 'in_Lab':
        img_RGB_new = color_transfer_in_Lab(img_RGB_source, img_RGB_target)
    elif option == 'in_CIECAM97s':
        img_RGB_new = color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target)
    return img_RGB_new


source_path = "images/source1.png"
target_path = "images/target1.png"
source = cv2.imread(source_path, cv2.IMREAD_COLOR)
source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
target = cv2.imread(target_path, cv2.IMREAD_COLOR)
target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
result = color_transfer(source, target, option='in_CIECAM97s')
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
imgplot = plt.imshow(result_rgb)
plt.show()


# if __name__ == "__main__":
#     print('==================================================')
#     print('PSU CS 410/510, Winter 2019, HW1: color transfer')
#     print('==================================================')
#
#     path_file_image_source = sys.argv[1]
#     path_file_image_target = sys.argv[2]
#     path_file_image_result_in_Lab = sys.argv[3]
#     path_file_image_result_in_RGB = sys.argv[4]
#     path_file_image_result_in_CIECAM97s = sys.argv[5]
#
#     # ===== read input images
#     # img_RGB_source: is the image you want to change the its color
#     # img_RGB_target: is the image containing the color distribution that you want to change the img_RGB_source to (transfer color of the img_RGB_target to the img_RGB_source)
#     img_RGB_source = cv2.imread(path_file_image_source, 1)
#     img_RGB_target = cv2.imread(path_file_image_target, 1)
#
#     img_RGB_new_Lab = color_transfer(img_RGB_source, img_RGB_target, option='in_Lab')
#     # todo: save image to path_file_image_result_in_Lab
#     cv2.imwrite(path_file_image_result_in_Lab, img_RGB_new_Lab)
#
#     img_RGB_new_RGB = color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')
#     # todo: save image to path_file_image_result_in_RGB
#     cv2.imwrite(path_file_image_result_in_RGB, img_RGB_new_RGB)
#
#     img_RGB_new_CIECAM97s = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')
#     # todo: save image to path_file_image_result_in_CIECAM97s
#     cv2.imwrite(path_file_image_result_in_CIECAM97s, img_RGB_new_CIECAM97s)