import cv2
import matplotlib.pyplot as plt
from color_transfer.color_transfer import ColourXfer

# 1) mean transfer using opencv and matrix colour space conversion on fig1(a) and fig1(B)
mean_fig1_a_path = "images/mean_fig1_a.png"
mean_fig1_a_bgr = cv2.imread(mean_fig1_a_path, cv2.IMREAD_COLOR)
mean_fig1_a_rgb = cv2.cvtColor(mean_fig1_a_bgr, cv2.COLOR_RGB2BGR)
mean_fig1_b_path = "images/mean_fig1_b.png"
mean_fig1_b_bgr = cv2.imread(mean_fig1_b_path, cv2.IMREAD_COLOR)
mean_fig1_b_rgb = cv2.cvtColor(mean_fig1_b_bgr, cv2.COLOR_RGB2BGR)
mean_fig1_c_opencv_path = "images/mean_fig1_c_opencv.png"
mean_fig1_c_matrix_path = "images/mean_fig1_c_matrix.png"
mean_fig1_c_opencv_rgb = ColourXfer(mean_fig1_a_rgb, mean_fig1_b_rgb, model='mean', conversion='opencv')
mean_fig1_c_matrix_rgb = ColourXfer(mean_fig1_a_rgb, mean_fig1_b_rgb, model='mean', conversion='matrix')
mean_fig1_c_opencv_bgr = cv2.cvtColor(mean_fig1_c_opencv_rgb, cv2.COLOR_RGB2BGR)
mean_fig1_c_matrix_bgr = cv2.cvtColor(mean_fig1_c_matrix_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(mean_fig1_c_opencv_path, mean_fig1_c_opencv_bgr)
cv2.imwrite(mean_fig1_c_matrix_path, mean_fig1_c_matrix_bgr)
mean_fig2_a_path = "images/mean_fig2_a.png"
mean_fig2_a_bgr = cv2.imread(mean_fig2_a_path, cv2.IMREAD_COLOR)
mean_fig2_a_rgb = cv2.cvtColor(mean_fig2_a_bgr, cv2.COLOR_RGB2BGR)
mean_fig2_b_path = "images/mean_fig2_b.png"
mean_fig2_b_bgr = cv2.imread(mean_fig2_b_path, cv2.IMREAD_COLOR)
mean_fig2_b_rgb = cv2.cvtColor(mean_fig2_b_bgr, cv2.COLOR_RGB2BGR)
mean_fig2_c_opencv_path = "images/mean_fig2_c_opencv.png"
mean_fig2_c_matrix_path = "images/mean_fig2_c_matrix.png"
mean_fig2_c_opencv_rgb = ColourXfer(mean_fig2_a_rgb, mean_fig2_b_rgb, model='mean', conversion='opencv')
mean_fig2_c_matrix_rgb = ColourXfer(mean_fig2_a_rgb, mean_fig2_b_rgb, model='mean', conversion='matrix')
mean_fig2_c_opencv_bgr = cv2.cvtColor(mean_fig2_c_opencv_rgb, cv2.COLOR_RGB2BGR)
mean_fig2_c_matrix_bgr = cv2.cvtColor(mean_fig2_c_matrix_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(mean_fig2_c_opencv_path, mean_fig2_c_opencv_bgr)
cv2.imwrite(mean_fig2_c_matrix_path, mean_fig2_c_matrix_bgr)
# plot result for figure 1
fig, ax = plt.subplots(2, 4, figsize=(18,6))
source_fig1 = plt.imread(mean_fig1_a_path)
ax[0][0].imshow(source_fig1)
ax[0][0].set_title('Source')
target_fig1 = plt.imread(mean_fig1_b_path)
ax[0][1].imshow(target_fig1)
ax[0][1].set_title('Target')
result_fig1_opencv = plt.imread(mean_fig1_c_opencv_path)
ax[0][2].imshow(result_fig1_opencv)
ax[0][2].set_title('OpenCV Result')
result_fig1_matrix = plt.imread(mean_fig1_c_matrix_path)
ax[0][3].imshow(result_fig1_matrix)
ax[0][3].set_title('Matrix Result')
# plot result for figure 2
source_fig2 = plt.imread(mean_fig2_a_path)
ax[1][0].imshow(source_fig2)
ax[1][0].set_title('Source')
target_fig2 = plt.imread(mean_fig2_b_path)
ax[1][1].imshow(target_fig2)
ax[1][1].set_title('Target')
result_fig2_opencv = plt.imread(mean_fig2_c_opencv_path)
ax[1][2].imshow(result_fig2_opencv)
ax[1][2].set_title('OpenCV Result')
result_fig2_matrix = plt.imread(mean_fig2_c_matrix_path)
ax[1][3].imshow(result_fig2_matrix)
ax[1][3].set_title('Matrix Result')
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
mean_failed_opencv_rgb = ColourXfer(pitie_source_rgb, pitie_target_rgb, model='mean', conversion='opencv')
mean_failed_matrix_rgb = ColourXfer(pitie_source_rgb, pitie_target_rgb, model='mean', conversion='matrix')
mean_failed_opencv_bgr = cv2.cvtColor(mean_failed_opencv_rgb, cv2.COLOR_RGB2BGR)
mean_failed_matrix_bgr = cv2.cvtColor(mean_failed_matrix_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(mean_failed_opencv_path, mean_failed_opencv_bgr)
cv2.imwrite(mean_failed_matrix_path, mean_failed_matrix_bgr)

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

# plot result all result
fig, ax = plt.subplots(2, 4, figsize=(18,6))
pitie_source = plt.imread(pitie_source_path)
ax[0][0].imshow(pitie_source)
ax[0][0].set_title('Source')
pitie_target = plt.imread(pitie_target_path)
ax[0][1].imshow(pitie_target)
ax[0][1].set_title('Target')
mean_failed_opencv = plt.imread(mean_failed_opencv_path)
ax[0][2].imshow(mean_failed_opencv)
ax[0][2].set_title('Failed Reinhard OpenCV Result')
mean_failed_matrix = plt.imread(mean_failed_matrix_path)
ax[0][3].imshow(mean_failed_matrix)
ax[0][3].set_title('Failed Reinhard Matrix Result')
pitie_idt_result = plt.imread(pitie_idt_result_path)
ax[1][0].imshow(pitie_idt_result)
ax[1][0].set_title('IDT Result')
pitie_regrain_result = plt.imread(pitie_regrain_result_path)
ax[1][1].imshow(pitie_regrain_result)
ax[1][1].set_title('IDT + Regain Result')
pitie_mkl_result = plt.imread(pitie_mkl_result_path)
ax[1][2].imshow(pitie_mkl_result)
ax[1][2].set_title('MKL Result')
fig.delaxes(ax[1][3])
plt.show()

# ************* time took ******************
# t = time.time()
# print("took {} s to be remove after testing phase".format(time.time() - t))



