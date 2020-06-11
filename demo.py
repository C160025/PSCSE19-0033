import numpy as np
import random
import scipy.optimize
from pprint import pprint
import scipy
import scipy.linalg
import cv2
import math
import time
from color_transfer.color_transfer import ColorXfer
from color_transfer.utils import cx_rgb2lab, cx_lab2rgb
import matplotlib.pyplot as plt
import pandas as pd

source_path = "images/source1.png"
target_path = "images/target1.png"
transfer_path = "images/source1_target1_opencv.png"
transfer_path1 = "images/source1_target1_matrix.png"
source_path1 = "images/source2.png"
target_path1 = "images/target2.png"
transfer_path2 = "images/source2_target2_opencv.png"
transfer_path3 = "images/source2_target2_matrix.png"
source_path2 = "images/scotland_house.png"
target_path2 = "images/scotland_plain.png"
transfer_path4 = "images/scotland_house_scotland_opencv.png"
transfer_path5 = "images/scotland_house_scotland_matrix.png"
transfer_path6 = "images/scotland_house_scotland_idt.png"
transfer_path7 = "images/scotland_house_scotland_regrain.png"
transfer_path8 = "images/scotland_house_scotland_mkl.png"


# 1) testing on mean opencv and matrix on source1 and traget1 result on matrix match on paper
# source_bgr = cv2.imread(source_path, cv2.IMREAD_COLOR)
# source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_RGB2BGR)
# target_bgr = cv2.imread(target_path, cv2.IMREAD_COLOR)
# target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_RGB2BGR)
# transfer_rgb = ColorXfer(source_rgb, target_rgb, model='mean', conversion='opencv')
# transfer_rgb1 = ColorXfer(source_rgb, target_rgb, model='mean', conversion='matrix')
# transfer_bgr = cv2.cvtColor(transfer_rgb, cv2.COLOR_RGB2BGR)
# transfer_bgr1 = cv2.cvtColor(transfer_rgb1, cv2.COLOR_RGB2BGR)
# cv2.imwrite(transfer_path, transfer_bgr)
# cv2.imwrite(transfer_path1, transfer_bgr1)

# 2) testing on mean opencv and matrix on source2 and traget2 result on matrix match on paper
# source_bgr1 = cv2.imread(source_path1, cv2.IMREAD_COLOR)
# source_rgb1 = cv2.cvtColor(source_bgr1, cv2.COLOR_RGB2BGR)
# target_bgr1 = cv2.imread(target_path1, cv2.IMREAD_COLOR)
# target_rgb1 = cv2.cvtColor(target_bgr1, cv2.COLOR_RGB2BGR)
# transfer_rgb2 = ColorXfer(source_rgb1, target_rgb1, model='mean', conversion='opencv')
# transfer_rgb3 = ColorXfer(source_rgb1, target_rgb1, model='mean', conversion='matrix')
# transfer_bgr2 = cv2.cvtColor(transfer_rgb2, cv2.COLOR_RGB2BGR)
# transfer_bgr3 = cv2.cvtColor(transfer_rgb3, cv2.COLOR_RGB2BGR)
# cv2.imwrite(transfer_path2, transfer_bgr2)
# cv2.imwrite(transfer_path3, transfer_bgr3)3

# 3) testing on mean opencv and matrix scotland_house and scotland_plain result failed on both
# source_bgr2 = cv2.imread(source_path2, cv2.IMREAD_COLOR)
# source_rgb2 = cv2.cvtColor(source_bgr2, cv2.COLOR_RGB2BGR)
# target_bgr2 = cv2.imread(target_path2, cv2.IMREAD_COLOR)
# target_rgb2 = cv2.cvtColor(target_bgr2, cv2.COLOR_RGB2BGR)
# transfer_rgb4 = ColorXfer(source_rgb2, target_rgb2, model='mean', conversion='opencv')
# transfer_rgb5 = ColorXfer(source_rgb2, target_rgb2, model='mean', conversion='matrix')
# transfer_bgr4 = cv2.cvtColor(transfer_rgb4, cv2.COLOR_RGB2BGR)
# transfer_bgr5 = cv2.cvtColor(transfer_rgb5, cv2.COLOR_RGB2BGR)
# cv2.imwrite(transfer_path4, transfer_bgr4)
# cv2.imwrite(transfer_path5, transfer_bgr5)

# 4) testing on idt matlab (F. Pitié) and recode on scotland_house and scotland_plain
# source_bgr2 = cv2.imread(source_path2, cv2.IMREAD_COLOR)
# source_rgb2 = cv2.cvtColor(source_bgr2, cv2.COLOR_RGB2BGR)
# target_bgr2 = cv2.imread(target_path2, cv2.IMREAD_COLOR)
# target_rgb2 = cv2.cvtColor(target_bgr2, cv2.COLOR_RGB2BGR)
# transfer_rgb6 = ColorXfer(source_rgb2, target_rgb2, model='idt')
# transfer_bgr6 = cv2.cvtColor(transfer_rgb6, cv2.COLOR_RGB2BGR)
# cv2.imwrite(transfer_path6, transfer_bgr6)

# 5) testing on idt + regrain matlab (F. Pitié) and recode on scotland_house and scotland_plain
# source_bgr2 = cv2.imread(source_path2, cv2.IMREAD_COLOR)
# source_rgb2 = cv2.cvtColor(source_bgr2, cv2.COLOR_RGB2BGR)
# target_bgr2 = cv2.imread(target_path2, cv2.IMREAD_COLOR)
# target_rgb2 = cv2.cvtColor(target_bgr2, cv2.COLOR_RGB2BGR)
# transfer_rgb7 = ColorXfer(source_rgb2, target_rgb2, model='regrain')
# transfer_bgr7 = cv2.cvtColor(transfer_rgb7, cv2.COLOR_RGB2BGR)
# cv2.imwrite(transfer_path7, transfer_bgr7)

# 6) testing on mkl matlab (F. Pitié) and recode on scotland_house and scotland_plain
source_bgr2 = cv2.imread(source_path2, cv2.IMREAD_COLOR)
source_rgb2 = cv2.cvtColor(source_bgr2, cv2.COLOR_RGB2BGR)
target_bgr2 = cv2.imread(target_path2, cv2.IMREAD_COLOR)
target_rgb2 = cv2.cvtColor(target_bgr2, cv2.COLOR_RGB2BGR)
transfer_rgb8 = ColorXfer(source_rgb2, target_rgb2, model='mkl')
transfer_bgr8 = cv2.cvtColor(transfer_rgb8, cv2.COLOR_RGB2BGR)
cv2.imwrite(transfer_path8, transfer_bgr8)


# ************* time took ******************
# t = time.time()
# print("took {} s to be remove after testing phase".format(time.time() - t))



