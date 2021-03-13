import cv2
import matplotlib.pyplot as plt
from matplotlib import ticker
from color_transfer.color_transfer import ColourXfer
import os
import ntpath


def genr_image_result(source, target):

    # 1) Mean model with matrix, no conversion and opencv colour spaces
    source_bgr = cv2.imread(source, cv2.IMREAD_COLOR)
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_RGB2BGR)
    filename = os.path.splitext(os.path.basename(source))[0]
    target_bgr = cv2.imread(target, cv2.IMREAD_COLOR)
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_RGB2BGR)
    mean_matrix_path = "images/" + filename + "_mean_matrix.png"
    mean_noconv_path = "images/" + filename + "_mean_noconv.png"
    mean_opencv_path = "images/" + filename + "_mean_opencv.png"
    mean_matrix_rgb = ColourXfer(source_rgb, target_rgb, model='mean', conversion='matrix')
    mean_noconv_rgb = ColourXfer(source_rgb, target_rgb, model='mean', conversion='noconv')
    mean_opencv_rgb = ColourXfer(source_rgb, target_rgb, model='mean', conversion='opencv')
    mean_matrix_bgr = cv2.cvtColor(mean_matrix_rgb, cv2.COLOR_RGB2BGR)
    mean_noconv_bgr = cv2.cvtColor(mean_noconv_rgb, cv2.COLOR_RGB2BGR)
    mean_opencv_bgr = cv2.cvtColor(mean_opencv_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(mean_matrix_path, mean_matrix_bgr)
    cv2.imwrite(mean_noconv_path, mean_noconv_bgr)
    cv2.imwrite(mean_opencv_path, mean_opencv_bgr)

    # 2) IDT model
    idt_result_path = "images/"+ filename + "_idt_result.png"
    idt_result_rgb = ColourXfer(source_rgb, target_rgb, model='idt')
    pitie_idt_result_bgr = cv2.cvtColor(idt_result_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(idt_result_path, pitie_idt_result_bgr)

    # 3) IDT + Regrain transfer
    regrain_result_path = "images/"+ filename + "_regrain_result.png"
    regrain_result_rgb = ColourXfer(source_rgb, target_rgb, model='regrain')
    regrain_result_bgr = cv2.cvtColor(regrain_result_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(regrain_result_path, regrain_result_bgr)

    # 4) MKL model
    mkl_result_path = "images/"+ filename + "_mkl_result.png"
    mkl_result_rgb = ColourXfer(source_rgb, target_rgb, model='mkl')
    mkl_result_bgr = cv2.cvtColor(mkl_result_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(mkl_result_path, mkl_result_bgr)

    # plot all result
    fig, ax = plt.subplots(3, 3, figsize=(18,13))
    image_source = plt.imread(source)
    ax[0][0].imshow(image_source)
    ax[0][0].set_title('Source')
    image_target = plt.imread(target)
    ax[0][1].imshow(image_target)
    ax[0][1].set_title('Target')
    mean_matrix_result = plt.imread(mean_matrix_path)
    ax[1][0].imshow(mean_matrix_result)
    ax[1][0].set_title('Mean Matrix Result')
    mean_noconv_result = plt.imread(mean_noconv_path)
    ax[1][1].imshow(mean_noconv_result)
    ax[1][1].set_title('Mean No Conversion Result')
    mean_opencv_result = plt.imread(mean_opencv_path)
    ax[1][2].imshow(mean_opencv_result)
    ax[1][2].set_title('Mean OpenCV Result')
    idt_result = plt.imread(idt_result_path)
    ax[2][0].imshow(idt_result)
    ax[2][0].set_title('IDT Result')
    regrain_result = plt.imread(regrain_result_path)
    ax[2][1].imshow(regrain_result)
    ax[2][1].set_title('IDT + Regain Result')
    mkl_result = plt.imread(mkl_result_path)
    ax[2][2].imshow(mkl_result)
    ax[2][2].set_title('MKL Result')
    fig.delaxes(ax[0][2])
    plt.show()


def save_color_histogram(path):
    '''
    RGB colour histogram
    :param path: image file path
    :return: show histogram graph
    '''
    image = cv2.imread(path, -1)
    path, filename = ntpath.split(path)
    for channel, col in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])

    plt.title(f'Colour Histogram for {filename}')
    plt.savefig(path + '/histograms_' + filename)
    plt.close()

def color_histogram(ax, path):
    '''
    RGB colour histogram
    :param path: image file path
    :return: histogram graph
    '''
    image = cv2.imread(path, -1)
    _, filename = ntpath.split(path)
    for channel, col in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
        ax.plot(hist, color=col)
        ax.set_xlim([0, 256])

def genr_1d_result(source, target):

    plt.close('all')
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(18, 13))
    color_histogram(ax1, source)
    ax1.set_title('Source')
    color_histogram(ax2, target)
    ax2.set_title('Target')
    filename = os.path.splitext(os.path.basename(source))[0]
    mean_matrix_path = "images/" + filename + "_mean_matrix.png"
    color_histogram(ax4, mean_matrix_path)
    ax4.set_title('Mean Matrix Result')
    mean_noconv_path = "images/" + filename + "_mean_noconv.png"
    color_histogram(ax5, mean_noconv_path)
    ax5.set_title('Mean No Conversion Result')
    mean_opencv_path = "images/" + filename + "_mean_opencv.png"
    color_histogram(ax6, mean_opencv_path)
    ax6.set_title('Mean OpenCV Result')
    idt_result_path = "images/" + filename + "_idt_result.png"
    color_histogram(ax7, idt_result_path)
    ax7.set_title('IDT Result')
    regrain_result_path = "images/" + filename + "_regrain_result.png"
    color_histogram(ax8, regrain_result_path)
    ax8.set_title('IDT + Regain Result')
    mkl_result_path = "images/" + filename + "_mkl_result.png"
    color_histogram(ax9, mkl_result_path)
    ax9.set_title('MKL Result')
    fig.delaxes(ax3)
    plt.show()
    plt.close()
    save_color_histogram(target)
    save_color_histogram(source)
    save_color_histogram(mean_matrix_path)
    save_color_histogram(mean_noconv_path)
    save_color_histogram(mean_opencv_path)
    save_color_histogram(idt_result_path)
    save_color_histogram(regrain_result_path)
    save_color_histogram(mkl_result_path)

def save_all_2d_histogram(path, bins=24, tick_spacing=2):
    '''
    2D colour histogram
    :param path: image file path
    :param bins: bins size
    :param tick_spacing: x and y axes intervals
    :return: show histogram graph
    '''
    image = cv2.imread(path)
    path, filename = ntpath.split(path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.8))
    channels_mapping = {0: 'B', 1: 'G', 2: 'R'}
    for i, channels in enumerate([[0, 1], [0, 2], [1, 2]]):
        hist = cv2.calcHist([image], channels, None, [bins] * 2, [0, 256] * 2)

        channel_x = channels_mapping[channels[0]]
        channel_y = channels_mapping[channels[1]]

        ax = axes[i]
        ax.set_xlim([0, bins - 1])
        ax.set_ylim([0, bins - 1])

        ax.set_xlabel(f'Channel {channel_x}')
        ax.set_ylabel(f'Channel {channel_y}')
        ax.set_title(f'2D Colour Histogram for {channel_x} and '
                     f'{channel_y}')

        ax.yaxis.set_major_locator(
            ticker.MultipleLocator(tick_spacing))
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(tick_spacing))

        im = ax.imshow(hist)

    cbar_ax = fig.add_axes([0.87, 0.275, 0.015, 0.44])
    fig.subplots_adjust(right=0.85)
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle(f'2D Colour Histograms for {filename} with {bins} bins', fontsize=16, y=0.92)
    fig.savefig(path + '/2d_histograms_'+filename, dpi=fig.dpi)
    plt.close()

def gr_2d_histogram(ax, path, bins=24, tick_spacing=2):
    '''
    2D GR colour histogram
    :param path: image file path
    :param bins: bins size
    :param tick_spacing: x and y axes intervals
    :return: show histogram graph
    '''
    image = cv2.imread(path)
    _, filename = ntpath.split(path)
    channels_mapping = {0: 'B', 1: 'G', 2: 'R'}
    for i, channels in enumerate([[1, 2]]):
        hist = cv2.calcHist([image], channels, None, [bins] * 2, [0, 256] * 2)

        channel_x = channels_mapping[channels[0]]
        channel_y = channels_mapping[channels[1]]

        ax.set_xlim([0, bins - 1])
        ax.set_ylim([0, bins - 1])

        ax.set_xlabel(f'Channel {channel_x}')
        ax.set_ylabel(f'Channel {channel_y}')

        ax.yaxis.set_major_locator(
            ticker.MultipleLocator(tick_spacing))
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(tick_spacing))

        im = ax.imshow(hist)

    return im

def genr_2d_result(source, target):
    plt.close('all')
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(18, 15))
    im1 = gr_2d_histogram(ax1, source)
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('Source')
    im2 = gr_2d_histogram(ax2, target)
    fig.colorbar(im2, ax=ax2)
    ax2.set_title('Target')
    filename = os.path.splitext(os.path.basename(source))[0]
    mean_matrix_path = "images/" + filename + "_mean_matrix.png"
    im4 = gr_2d_histogram(ax4, mean_matrix_path)
    fig.colorbar(im4, ax=ax4)
    ax4.set_title('Mean Matrix Result')
    mean_noconv_path = "images/" + filename + "_mean_noconv.png"
    im5 = gr_2d_histogram(ax5, mean_noconv_path)
    fig.colorbar(im5, ax=ax5)
    ax5.set_title('Mean No Conversion Result')
    mean_opencv_path = "images/" + filename + "_mean_opencv.png"
    im6 = gr_2d_histogram(ax6, mean_opencv_path)
    fig.colorbar(im6, ax=ax6)
    ax6.set_title('Mean OpenCV Result')
    idt_result_path = "images/" + filename + "_idt_result.png"
    im7 = gr_2d_histogram(ax7, idt_result_path)
    fig.colorbar(im7, ax=ax7)
    ax7.set_title('IDT Result')
    regrain_result_path = "images/" + filename + "_regrain_result.png"
    im8 = gr_2d_histogram(ax8, regrain_result_path)
    fig.colorbar(im8, ax=ax8)
    ax8.set_title('IDT + Regain Result')
    mkl_result_path = "images/" + filename + "_mkl_result.png"
    im9 = gr_2d_histogram(ax9, mkl_result_path)
    fig.colorbar(im9, ax=ax9)
    ax9.set_title('MKL Result')
    fig.delaxes(ax3)
    plt.show()
    plt.close()
    save_all_2d_histogram(target)
    save_all_2d_histogram(source)
    save_all_2d_histogram(mean_matrix_path)
    save_all_2d_histogram(mean_noconv_path)
    save_all_2d_histogram(mean_opencv_path)
    save_all_2d_histogram(idt_result_path)
    save_all_2d_histogram(regrain_result_path)
    save_all_2d_histogram(mkl_result_path)

def genr_idt_iter_result():
    for n in range(30):
        file_path = "images/idt_iteration_images/idt_" + str(n+1) + "_iteration.png"
        save_color_histogram(file_path)
        save_all_2d_histogram(file_path)
    rows, cols, i, j, k = 3, 5, 0, 0, 0
    fig1, ax1 = plt.subplots(rows, cols, figsize=(29, 13))
    for r in range(rows):
        for c in range(cols):
            i = i + 2
            load_path = "images/idt_iteration_images/idt_" + str(i) + "_iteration.png"
            image = plt.imread(load_path)
            ax1[r][c].imshow(image)
            ax1[r][c].set_title(str(i) + " iteration")
    plt.show()
    fig2, ax2 = plt.subplots(rows, cols, figsize=(29, 13))
    for r in range(rows):
        for c in range(cols):
            j = j + 2
            load_path = "images/idt_iteration_images/idt_" + str(j) + "_iteration.png"
            color_histogram(ax2[r][c], load_path)
            ax2[r][c].set_title(str(j) + " iteration")
    plt.show()
    fig3, ax3 = plt.subplots(rows, cols, figsize=(29, 13))
    for r in range(rows):
        for c in range(cols):
            k = k + 2
            load_path = "images/idt_iteration_images/idt_" + str(k) + "_iteration.png"
            im3 = gr_2d_histogram(ax3[r][c], load_path)
            fig3.colorbar(im3, ax=ax3[r][c])
            ax3[r][c].set_title(str(k) + " iteration")
    plt.show()
    plt.close()

