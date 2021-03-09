import cv2
import matplotlib.pyplot as plt
from color_transfer.color_transfer import ColourXfer
from color_transfer.utils import cx_rgb2lab, cx_lab2rgb
import numpy as np


# 1) mean transfer using opencv and matrix colour space conversion on fig1(a) and fig1(B)
mean_fig1_a_path = "images/mean_fig1_a.png"
mean_fig1_a_bgr = cv2.imread(mean_fig1_a_path, cv2.IMREAD_COLOR)
mean_fig1_a_rgb = cv2.cvtColor(mean_fig1_a_bgr, cv2.COLOR_RGB2BGR)
mean_fig1_b_path = "images/mean_fig1_b.png"
mean_fig1_b_bgr = cv2.imread(mean_fig1_b_path, cv2.IMREAD_COLOR)
mean_fig1_b_rgb = cv2.cvtColor(mean_fig1_b_bgr, cv2.COLOR_RGB2BGR)
mean_fig1_c_opencv_path = "images/mean_fig1_c_opencv.png"
mean_fig1_c_matrix_path = "images/mean_fig1_c_matrix.png"
mean_fig1_c_noconv_path = "images/mean_fig1_c_noconv.png"
mean_fig1_c_opencv_rgb = ColourXfer(mean_fig1_a_rgb, mean_fig1_b_rgb, model='mean', conversion='opencv')
mean_fig1_c_matrix_rgb = ColourXfer(mean_fig1_a_rgb, mean_fig1_b_rgb, model='mean', conversion='matrix')
mean_fig1_c_noconv_rgb = ColourXfer(mean_fig1_a_rgb, mean_fig1_b_rgb, model='mean', conversion='noconv')
mean_fig1_c_opencv_bgr = cv2.cvtColor(mean_fig1_c_opencv_rgb, cv2.COLOR_RGB2BGR)
mean_fig1_c_matrix_bgr = cv2.cvtColor(mean_fig1_c_matrix_rgb, cv2.COLOR_RGB2BGR)
mean_fig1_c_noconv_bgr = cv2.cvtColor(mean_fig1_c_noconv_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(mean_fig1_c_opencv_path, mean_fig1_c_opencv_bgr)
cv2.imwrite(mean_fig1_c_matrix_path, mean_fig1_c_matrix_bgr)
cv2.imwrite(mean_fig1_c_noconv_path, mean_fig1_c_noconv_bgr)
mean_fig2_a_path = "images/mean_fig2_a.png"
mean_fig2_a_bgr = cv2.imread(mean_fig2_a_path, cv2.IMREAD_COLOR)
mean_fig2_a_rgb = cv2.cvtColor(mean_fig2_a_bgr, cv2.COLOR_RGB2BGR)
mean_fig2_b_path = "images/mean_fig2_b.png"
mean_fig2_b_bgr = cv2.imread(mean_fig2_b_path, cv2.IMREAD_COLOR)
mean_fig2_b_rgb = cv2.cvtColor(mean_fig2_b_bgr, cv2.COLOR_RGB2BGR)
mean_fig2_c_opencv_path = "images/mean_fig2_c_opencv.png"
mean_fig2_c_matrix_path = "images/mean_fig2_c_matrix.png"
mean_fig2_c_noconv_path = "images/mean_fig2_c_noconv.png"
mean_fig2_c_opencv_rgb = ColourXfer(mean_fig2_a_rgb, mean_fig2_b_rgb, model='mean', conversion='opencv')
mean_fig2_c_matrix_rgb = ColourXfer(mean_fig2_a_rgb, mean_fig2_b_rgb, model='mean', conversion='matrix')
mean_fig2_c_noconv_rgb = ColourXfer(mean_fig2_a_rgb, mean_fig2_b_rgb, model='mean', conversion='noconv')
mean_fig2_c_opencv_bgr = cv2.cvtColor(mean_fig2_c_opencv_rgb, cv2.COLOR_RGB2BGR)
mean_fig2_c_matrix_bgr = cv2.cvtColor(mean_fig2_c_matrix_rgb, cv2.COLOR_RGB2BGR)
mean_fig2_c_noconv_bgr = cv2.cvtColor(mean_fig2_c_noconv_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(mean_fig2_c_opencv_path, mean_fig2_c_opencv_bgr)
cv2.imwrite(mean_fig2_c_matrix_path, mean_fig2_c_matrix_bgr)
cv2.imwrite(mean_fig2_c_noconv_path, mean_fig2_c_noconv_bgr)
# plot result for figure 1
fig, ax = plt.subplots(4, 3, figsize=(18,16))
source_fig1 = plt.imread(mean_fig1_a_path)
ax[0][0].imshow(source_fig1)
ax[0][0].set_title('Source')
target_fig1 = plt.imread(mean_fig1_b_path)
ax[0][1].imshow(target_fig1)
ax[0][1].set_title('Target')
result_fig1_opencv = plt.imread(mean_fig1_c_opencv_path)
ax[1][0].imshow(result_fig1_opencv)
ax[1][0].set_title('OpenCV Result')
result_fig1_matrix = plt.imread(mean_fig1_c_matrix_path)
ax[1][1].imshow(result_fig1_matrix)
ax[1][1].set_title('Matrix Result')
result_fig1_noconv = plt.imread(mean_fig1_c_noconv_path)
ax[1][2].imshow(result_fig1_noconv)
ax[1][2].set_title('No Conversion Result')
# plot result for figure 2
source_fig2 = plt.imread(mean_fig2_a_path)
ax[2][0].imshow(source_fig2)
ax[2][0].set_title('Source')
target_fig2 = plt.imread(mean_fig2_b_path)
ax[2][1].imshow(target_fig2)
ax[2][1].set_title('Target')
result_fig2_opencv = plt.imread(mean_fig2_c_opencv_path)
ax[3][0].imshow(result_fig2_opencv)
ax[3][0].set_title('OpenCV Result')
result_fig2_matrix = plt.imread(mean_fig2_c_matrix_path)
ax[3][1].imshow(result_fig2_matrix)
ax[3][1].set_title('Matrix Result')
result_fig2_noconv = plt.imread(mean_fig2_c_noconv_path)
ax[3][2].imshow(result_fig2_noconv)
ax[3][2].set_title('No Conversion Result')
fig.delaxes(ax[0][2])
fig.delaxes(ax[2][2])
plt.show()

# 2) mean transfer using opencv and matrix colour space conversion on Pitié's source and Pitié's target failed on Reinhard transfer
pitie_source_path = "images/pitie_source.png"
pitie_source_bgr = cv2.imread(pitie_source_path, cv2.IMREAD_COLOR)
pitie_source_rgb = cv2.cvtColor(pitie_source_bgr, cv2.COLOR_RGB2BGR)
pitie_target_path = "images/pitie_target.png"
pitie_target_bgr = cv2.imread(pitie_target_path, cv2.IMREAD_COLOR)
pitie_target_rgb = cv2.cvtColor(pitie_target_bgr, cv2.COLOR_RGB2BGR)
mean_failed_opencv_path = "images/mean_failed_opencv.png"
mean_failed_matrix_path = "images/mean_failed_matrix.png"
mean_failed_noconv_path = "images/mean_failed_noconv.png"
mean_failed_opencv_rgb = ColourXfer(pitie_source_rgb, pitie_target_rgb, model='mean', conversion='opencv')
mean_failed_matrix_rgb = ColourXfer(pitie_source_rgb, pitie_target_rgb, model='mean', conversion='matrix')
mean_failed_noconv_rgb = ColourXfer(pitie_source_rgb, pitie_target_rgb, model='mean', conversion='noconv')
mean_failed_opencv_bgr = cv2.cvtColor(mean_failed_opencv_rgb, cv2.COLOR_RGB2BGR)
mean_failed_matrix_bgr = cv2.cvtColor(mean_failed_matrix_rgb, cv2.COLOR_RGB2BGR)
mean_failed_noconv_bgr = cv2.cvtColor(mean_failed_noconv_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(mean_failed_opencv_path, mean_failed_opencv_bgr)
cv2.imwrite(mean_failed_matrix_path, mean_failed_matrix_bgr)
cv2.imwrite(mean_failed_noconv_path, mean_failed_noconv_bgr)

# 3) idt transfer on Pitié's source and Pitié's target
pitie_idt_result_path = "images/pitie_idt_result.png"
pitie_idt_result_rgb = ColourXfer(pitie_source_rgb, pitie_target_rgb, model='idt')
pitie_idt_result_bgr = cv2.cvtColor(pitie_idt_result_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(pitie_idt_result_path, pitie_idt_result_bgr)

# 4) idt + regain transfer on Pitié's source and Pitié's target
pitie_regrain_result_path = "images/pitie_regrain_result.png"
pitie_regrain_result_rgb = ColourXfer(pitie_source_rgb, pitie_target_rgb, model='regrain')
pitie_regrain_result_bgr = cv2.cvtColor(pitie_regrain_result_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(pitie_regrain_result_path, pitie_regrain_result_bgr)

# 4) mkl transfer on Pitié's source and Pitié's target
pitie_mkl_result_path = "images/pitie_mkl_result.png"
pitie_mkl_result_rgb = ColourXfer(pitie_source_rgb, pitie_target_rgb, model='mkl')
pitie_mkl_result_bgr = cv2.cvtColor(pitie_mkl_result_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(pitie_mkl_result_path, pitie_mkl_result_bgr)

# 5) mkl transfer on Pitié's source and Pitié's target
pitie_mkl_result_lab_path = "images/pitie_mkl_result_lab.png"
pitie_source_lab = cv2.cvtColor(pitie_source_rgb, cv2.COLOR_RGB2LAB).astype(np.float)
pitie_target_lab = cv2.cvtColor(pitie_target_rgb, cv2.COLOR_RGB2LAB).astype(np.float)
# pitie_source_lab = cx_rgb2lab(pitie_source_rgb, True)
# pitie_target_lab = cx_rgb2lab(pitie_target_rgb, True)
pitie_mkl_result_lab = ColourXfer(pitie_source_lab, pitie_target_lab, model='mkl')
pitie_mkl_result_rgb = cv2.cvtColor(pitie_mkl_result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
# pitie_mkl_result_rgb = cx_lab2rgb(pitie_mkl_result_lab, True)
pitie_mkl_result_bgr = cv2.cvtColor(pitie_mkl_result_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(pitie_mkl_result_lab_path, pitie_mkl_result_bgr)
# mean_fig1_a = cx_rgb2lab(mean_fig1_a_rgb, True)
# mean_fig1_b = cx_lab2rgb(mean_fig1_a, True)

# plot result all result
fig, ax = plt.subplots(3, 3, figsize=(18,13))
pitie_source = plt.imread(pitie_source_path)
ax[0][0].imshow(pitie_source)
ax[0][0].set_title('Source')
pitie_target = plt.imread(pitie_target_path)
ax[0][1].imshow(pitie_target)
ax[0][1].set_title('Target')
mean_failed_opencv = plt.imread(mean_failed_opencv_path)
ax[1][0].imshow(mean_failed_opencv)
ax[1][0].set_title('Failed Reinhard OpenCV Result')
mean_failed_matrix = plt.imread(mean_failed_matrix_path)
ax[1][1].imshow(mean_failed_matrix)
ax[1][1].set_title('Failed Reinhard Matrix Result')
mean_failed_noconv = plt.imread(mean_failed_noconv_path)
ax[1][2].imshow(mean_failed_noconv)
ax[1][2].set_title('Failed Reinhard No Conversion Result')
pitie_idt_result = plt.imread(pitie_idt_result_path)
ax[2][0].imshow(pitie_idt_result)
ax[2][0].set_title('IDT Result')
pitie_regrain_result = plt.imread(pitie_regrain_result_path)
ax[2][1].imshow(pitie_regrain_result)
ax[2][1].set_title('IDT + Regain Result')
pitie_mkl_result = plt.imread(pitie_mkl_result_lab_path)
ax[2][2].imshow(pitie_mkl_result)
ax[2][2].set_title('MKL Result')
fig.delaxes(ax[0][2])
plt.show()
#
# # ************* time took ******************
# # t = time.time()
# # print("took {} s to be remove after testing phase".format(time.time() - t))
#
#
#

# rgb_histo_path = "images/mean_fig2_a.png"
# mean_fig2_a_bgr = cv2.imread(mean_fig2_a_path, cv2.IMREAD_COLOR)
# mean_fig1_a_rgb = cv2.cvtColor(mean_fig1_a_bgr, cv2.COLOR_RGB2BGR)
# mean_fig1_a = cx_rgb2lab(mean_fig1_a_rgb, True)
# mean_fig1_b = cx_lab2rgb(mean_fig1_a, True)
# result_path = "images/resultMatrix_fig1_a.png"
# mean_fig2_b_rgb = cv2.cvtColor(mean_fig1_b, cv2.COLOR_RGB2BGR)
# cv2.imwrite(result_path, mean_fig2_b_rgb)
