import cv2
import ntpath
from matplotlib import pyplot as plt
from matplotlib import ticker


def show_color_histogram(path):
    '''
    RGB colour histogram
    :param path: image file path
    :return: show histogram graph
    '''
    image = cv2.imread(path, -1)
    _, filename = ntpath.split(path)
    for channel, col in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])

    plt.title(f'Colour Histogram for {filename}')
    plt.savefig('images/histograms_' + filename)
    plt.show()
    plt.close()

def show_image_histogram_2d(path, bins=16, tick_spacing=2):
    '''
    2D colour histogram
    :param path: image file path
    :param bins: bins size
    :param tick_spacing: x and y axes intervals
    :return: show histogram graph
    '''
    image = cv2.imread(path)
    _, filename = ntpath.split(path)
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
    fig.savefig('images/2d_histograms_'+filename, dpi=fig.dpi)
    plt.show()
    plt.close()

# histograms for Reinhard
mean_fig1 = 'images/mean_fig1_a.png'
show_color_histogram(mean_fig1)
mean_fig1_b = 'images/mean_fig1_b.png'
show_color_histogram(mean_fig1_b)
mean_fig1_c_matrix = 'images/mean_fig1_c_matrix.png'
show_color_histogram(mean_fig1_c_matrix)
mean_fig1_c_noconv = 'images/mean_fig1_c_noconv.png'
show_color_histogram(mean_fig1_c_noconv)
mean_fig1_c_opencv = 'images/mean_fig1_c_opencv.png'
show_color_histogram(mean_fig1_c_opencv)

# 2D histograms for Reinhard
mean_fig1 = 'images/mean_fig1_a.png'
show_image_histogram_2d(mean_fig1)
mean_fig1_b = 'images/mean_fig1_b.png'
show_image_histogram_2d(mean_fig1_b)
mean_fig1_c_matrix = 'images/mean_fig1_c_matrix.png'
show_image_histogram_2d(mean_fig1_c_matrix)
mean_fig1_c_noconv = 'images/mean_fig1_c_noconv.png'
show_image_histogram_2d(mean_fig1_c_noconv)
mean_fig1_c_opencv = 'images/mean_fig1_c_opencv.png'
show_image_histogram_2d(mean_fig1_c_opencv)

# histograms for Reinhard 2
mean_fig2 = 'images/mean_fig2_a.png'
show_color_histogram(mean_fig2)
mean_fig2_b = 'images/mean_fig2_b.png'
show_color_histogram(mean_fig2_b)
mean_fig2_c_matrix = 'images/mean_fig2_c_matrix.png'
show_color_histogram(mean_fig2_c_matrix)
mean_fig2_c_noconv = 'images/mean_fig2_c_noconv.png'
show_color_histogram(mean_fig2_c_noconv)
mean_fig2_c_opencv = 'images/mean_fig2_c_opencv.png'
show_color_histogram(mean_fig2_c_opencv)

# 2D histograms for Reinhard 2
mean_fig2 = 'images/mean_fig2_a.png'
show_image_histogram_2d(mean_fig2)
mean_fig2_b = 'images/mean_fig2_b.png'
show_image_histogram_2d(mean_fig2_b)
mean_fig2_c_matrix = 'images/mean_fig2_c_matrix.png'
show_image_histogram_2d(mean_fig2_c_matrix)
mean_fig2_c_noconv = 'images/mean_fig2_c_noconv.png'
show_image_histogram_2d(mean_fig2_c_noconv)
mean_fig2_c_opencv = 'images/mean_fig2_c_opencv.png'
show_image_histogram_2d(mean_fig2_c_opencv)

# histograms for Pitie
pitie_source = 'images/pitie_source.png'
show_color_histogram(pitie_source)
pitie_target = 'images/pitie_target.png'
show_color_histogram(pitie_target)
mean_failed = 'images/mean_failed_matrix.png'
show_color_histogram(mean_failed)
idt_result = 'images/pitie_idt_result.png'
show_color_histogram(idt_result)
regrain_result = 'images/pitie_regrain_result.png'
show_color_histogram(regrain_result)
mkl_result = 'images/pitie_mkl_result.png'
show_color_histogram(mkl_result)

# 2D histograms for Pitie
pitie_source = 'images/pitie_source.png'
show_image_histogram_2d(pitie_source)
pitie_target = 'images/pitie_target.png'
show_image_histogram_2d(pitie_target)
mean_failed = 'images/mean_failed_matrix.png'
show_image_histogram_2d(mean_failed)
idt_result = 'images/pitie_idt_result.png'
show_image_histogram_2d(idt_result)
regrain_result = 'images/pitie_regrain_result.png'
show_image_histogram_2d(regrain_result)
mkl_result = 'images/pitie_mkl_result.png'
show_image_histogram_2d(mkl_result)

