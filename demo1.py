mean_fig1_b_path = "images/mean_fig1_b.png"

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import cv2
#
mean_fig1_b_bgr = cv2.imread(mean_fig1_b_path, cv2.IMREAD_COLOR)
mean_fig1_b_rgb = cv2.cvtColor(mean_fig1_b_bgr, cv2.COLOR_RGB2BGR)

# r, g, b = cv2.split(mean_fig1_b_rgb)
# fig = plt.figure()
# # axis = fig.add_subplot(1, 1, 1, projection="3d")
# axis = fig.gca(projection='3d')
#
# pixel_colors = mean_fig1_b_rgb.reshape((np.shape(mean_fig1_b_rgb)[0]*np.shape(mean_fig1_b_rgb)[1], 3))
# norm = colors.Normalize(vmin=-1.,vmax=1.)
# norm.autoscale(pixel_colors)
# pixel_colors = norm(pixel_colors).tolist()
#
#
# axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Red")
# axis.set_ylabel("Green")
# axis.set_zlabel("Blue")
# plt.show()

# axis.scatter(r, g, b, c="#ff0000", marker="o")
# axis.set_xlabel("Red")
# axis.set_ylabel("Green")
# axis.set_zlabel("Blue")
# pyplot.show()


# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# from matplotlib import cm
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X, Y, Z = cv2.split(mean_fig1_b_rgb)
# # X, Y, Z = axes3d.get_test_data(0.05)
# ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
# cset = ax.contourf(X, Y, Z, zdir='z', cmap=cm.coolwarm)
# cset = ax.contourf(X, Y, Z, zdir='x',  cmap=cm.coolwarm)
# cset = ax.contourf(X, Y, Z, zdir='y', cmap=cm.coolwarm)
#
# ax.set_xlabel('X')
# ax.set_xlim(-40, 40)
# ax.set_ylabel('Y')
# ax.set_ylim(-40, 40)
# ax.set_zlabel('Z')
# ax.set_zlim(-100, 100)
#
# plt.show()

import pandas as pd


im_orig = plt.imread('images/pitie_source.png')
im_target = plt.imread('images/pitie_target.png')
# im_result = plt.imread('images/pitie_mkl_result.png')
im_result = plt.imread('images/pitie_idt_result.png')
im_result = plt.imread('images/pitie_regrain_result.png')

df_orig = pd.DataFrame(im_orig.reshape(-1, im_orig.shape[-1]), columns=['r', 'g', 'b'])
df_target = pd.DataFrame(im_target.reshape(-1, im_target.shape[-1]), columns=['r', 'g', 'b'])

a_orig = df_orig.values
a_target = df_target.values
df_result = pd.DataFrame(im_result.reshape(-1, im_target.shape[-1]), columns=['r', 'g', 'b'])

a_result = df_result.values
im_result = a_result.reshape(im_orig.shape)

from itertools import combinations
import matplotlib as mpl

def my_colorplot(df_orig, df_target, df_result, bins=100, size=3):
    colors = zip(
        df_orig.columns[:-1],
        # df_orig.drop(columns=['g']),
        df_orig.columns[1:]
    )
    cols = [0, 2]
    h = df_orig.columns[:-1]
    l = df_orig.columns[1:]
    k = df_orig.drop(columns=['g'])
    k1 = pd.DataFrame(data=k)
    color_pairs = list(combinations(colors, 2))

    nrows = len(color_pairs)

    figsize = (3 * size, nrows * size)

    x_lim = np.empty((3, 2))
    y_lim = np.empty((3, 2))

    fig, ax = plt.subplots(nrows, 3, figsize=figsize)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    if nrows == 1:
        ax = [ax]

    for i, (a, b) in enumerate(color_pairs):
        ax[i][0].hist2d(
            df_orig[a[0]] - df_orig[a[1]],
            df_orig[b[0]] - df_orig[b[1]],
            bins=bins, norm=mpl.colors.LogNorm()
        )
        x_lim[0] = ax[i][0].get_xlim()
        y_lim[0] = ax[i][0].get_ylim()

        ax[i][1].hist2d(
            df_target[a[0]] - df_target[a[1]],
            df_target[b[0]] - df_target[b[1]],
            bins=bins, norm=mpl.colors.LogNorm()
        )
        x_lim[1] = ax[i][1].get_xlim()
        y_lim[1] = ax[i][1].get_ylim()

        ax[i][2].hist2d(
            df_result[a[0]] - df_result[a[1]],
            df_result[b[0]] - df_result[b[1]],
            bins=bins, norm=mpl.colors.LogNorm()
        )
        x_lim[2] = ax[i][2].get_xlim()
        y_lim[2] = ax[i][2].get_ylim()

        ax[i][0].set_ylabel('{0} - {1}'.format(b[0], b[1]))
        for j in range(3):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].set_xticklabels([])
            ax[i][j].set_yticklabels([])
            ax[i][j].set_xlim([x_lim[:2, 0].min(), x_lim[:2, 1].max()])
            ax[i][j].set_ylim([y_lim[:2, 0].min(), y_lim[:2, 1].max()])

        ax2 = ax[i][j].twinx()
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.set_ylabel('{0} - {1}'.format(a[0], a[1]))

    ax[0][0].set_title('orig')
    ax[0][1].set_title('target')
    ax[0][2].set_title('result')

bins = 100
my_colorplot(df_orig, df_target, df_result, bins=bins)


plt.show()