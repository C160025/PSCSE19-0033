import matplotlib
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

mm = lambda d: d/25.4

nplots = 2
wp, hp = mm(40), mm(28)
dxp, dyp = mm(16), mm(12)

nrows, ncols = 3, 2
wf, hf = nplots*(wp+dxp), hp+dyp
dxf, dyf = mm(10), mm(8)

xcorners, ycorners = (np.arange(dxf/2,ncols*(wf+dxf),wf+dxf),
                      np.arange(dyf/2,nrows*(hf+dyf),hf+dyf))

# plus 10 mm for suptitle
fig = plt.figure(figsize=(ncols*(wf+dxf), nrows*(hf+dyf)+mm(10)))

rect = lambda xy: plt.Rectangle(xy, wf, hf,
                                transform=fig.dpi_scale_trans,
                                figure=fig,
                                edgecolor='k', facecolor='none')
fig.patches.extend([rect(xy) for xy in product(xcorners, ycorners)])

t = np.linspace(0,3.14,315); s = np.sin(t)

for nframe, (y, x) in enumerate(product(ycorners, xcorners), 1):
    for n in range(nplots):
        divider = Divider(fig, (0.0, 0.0, 1., 1.),
                          [Size.Fixed(x+0.7*dxp+n*(wp+dxp)), Size.Fixed(wp)],
                          [Size.Fixed(y+0.7*dyp           ), Size.Fixed(hp)],
                          aspect=False)
        ax = Axes(fig, divider.get_position())
        ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
        ax.plot(t, s)
        fig.add_axes(ax)
        fig.text(x, y, 'Frame %d'%nframe, transform=fig.dpi_scale_trans)

figsize = fig.get_size_inches()
width = figsize[0]*25.4 # mm
fig.suptitle('Original figure width is %.2f mm - everything is scaled'%width)
fig.savefig('pippo.png', dpi=118, facecolor='#f8f8f0')


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