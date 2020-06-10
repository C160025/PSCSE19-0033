import numpy as np
from scipy.stats import special_ortho_group
import scipy as sp
import cv2

# x1 = np.arange(9.0).reshape((3, 3))
# print(x1)
# x2 = np.arange(3.0)
# print(x2)
# x4 = x1 / x2
# print(x4)
# x3 = np.divide(x1, x2)
# print(x3)
#nbins = 300
#xs = np.linspace(0, nbins - 1, num=nbins)
#x = np.linspace(0, 100-1, num=100)
#print(xs)

#a = np.random.random(size=(3, 3))
#q, _ = np.linalg.qr(a)
#print(q)
# nb_channels = 10
# rot = np.array([[1, 0, 0],
#               [0, 1, 0],
#               [0, 0, 1],
#               [2 / 3, 2 / 3, -1 / 3],
#               [2 / 3, -1 / 3, 2 / 3],
#               [-1 / 3, 2 / 3, 2 / 3]])
#
# rotation = np.zeros((nb_channels, rot.shape[0], rot.shape[1]))
#
# rotation[0, :] = rot
#
#
# for i in range(1, nb_channels):
#     q, r = np.linalg.qr(np.random.random(size=(3, 3)))
#     xx = np.matmul(rot, q)
#     rotation[i, :] = xx
# # # for _ in (number+1 for number in range(5)):
# print(rotation)
#vres = 9
#x = list(range(vres)) + [vres]
#print(x)
# arr1 = np.array([[0,  1,  2, 3],
#                  [4,  5, 6, 7],
#                  [8, 9, 10, 11]])
# arr2 = np.array([[0,  1,  2, 3, 4],
#                 [4,  5, 6, 7, 8],
#                 [8, 9, 10, 11, 12]])
# arr3 = np.array([[0, 1, 2],
#                  [3, 4, 5],
#                  [6, 7, 8],
#                  [9, 10, 11]])
# arr4 = np.array([[1, 0, 0],
#                  [0, 1, 0],
#                  [0, 0, 1],
#                  [0.666666666666667, 0.666666666666667, -0.333333333333333],
#                  [0.666666666666667, -0.333333333333333, 0.666666666666667],
#                  [-0.333333333333333, 0.666666666666667, 0.666666666666667]])
# arr5 = np.array([[-0.0946985255440515, -0.0946985255440515,	-0.0974588624518559, -0.0946985255440515, -0.0865754169406229, -0.0946985255440515, -0.100683206207696, -0.0974588624518559, -0.103878506859383, -0.103878506859383],
#                  [-0.148777619671240, -0.141825457958536, -0.148777619671240, -0.141825457958536, -0.135321857981029, -0.151916493125442, -0.155017924887804, -0.148777619671240, -0.151916493125442, -0.158084076720273],
#                  [-0.158642636105230, -0.160210997058362, -0.159751916983567, -0.158740523679379, -0.159733337877987, -0.161515388961310, -0.157574869725283, -0.158740523679379, -0.159751916983567, -0.159751916983567],
#                  [-0.0931168538848894, -0.0931168538848894,	-0.0960969899240176, -0.0951065028757699, -0.0843510492428852, -0.0970863573656762, -0.107643079883114, -0.100977049872590, -0.101933972572696, -0.105766328109750],
#                  [-0.138231552060470, -0.133768570727183, -0.138231552060470, -0.130114508398157, -0.128538351494821, -0.134566006817967, -0.128538351494821, -0.130114508398157, -0.140391315283744, -0.139021823485228],
#                  [-0.187898092194062, -0.183084895672678, -0.187727261137733, -0.180801658360903, -0.181355403340932, -0.187764975154194, -0.183084895672678, -0.182499459760807, -0.187727261137733, -0.188193254166890]])
# arr6 = np.array([[[1, 2, 3],
#                   [4, 5, 6],
#                   [7, 8, 9]],
#                  [[11, 12, 13],
#                   [14, 15, 16],
#                   [17, 18, 19]],
#                  [[21, 22, 23],
#                   [24, 25, 26],
#                   [27, 28, 29]]])
# arr = np.zeros((2, 2, 3))
# for i in range(3):
#     sss = arr1[i, :].reshape((2, 2))
#     arr[:, :, i] = sss
# print(arr)
#cv2.merge([l, a, b])
#tt = t.reshape(arr6.shape[-1], -1)
#print(tt)
#r = special_ortho_group.rvs(3).astype(np.float)
#rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [2 / 3, 2 / 3, -1 / 3], [2 / 3, -1 / 3, 2 / 3], [-1 / 3, 2 / 3, 2 / 3]])
#rotation = np.zeros((10, rot.shape[0], rot.shape[1]))
#rotation[0, :] = rot
#for i in range(1, 10):
#    r = special_ortho_group.rvs(3).astype(np.float)
#    # q, r = np.linalg.qr(np.random.random(size=(3, 3)))
#    # qt = r.T
#    t = rot.dot(r)
#    #tt = np.fabs(sp.linalg.det(t))
#    #print(tt)
#    rotation[i, :, ] = t

# q, r = np.linalg.qr(np.random.random(size=(3, 3)))
# qt = r.T
#ttt = np.dot(rot, r)
#rotation[i, :, ] = ttt
#xx = np.matmul(np.linalg.pinv(arr4), arr5)
#print(rotation)
# eps = np.finfo(np.float32).eps
# arr1 = np.array([[0, 1,  2, 3],
#                  [4,  5, 6, 7],
#                  [8, 9, 10, 11]])
# arr = np.array([[[0, 12, 24],
#                  [1, 13, 25],
#                  [2, 14, 26],
#                  [3, 15, 27]],
#                 [[4, 16, 28],
#                  [5, 17, 29],
#                  [6, 18, 30],
#                  [7, 19, 31]],
#                 [[8, 20, 32],
#                  [9, 21, 33],
#                  [10, 22, 34],
#                  [11, 23, 35]]])
# [dim, row, col] = arr.shape
# print('k=' + str(dim) + ' vres=' + str(row) + ' hres=' + str(col))
# level = 3
# smoothness = 1
#
# p1 = lambda arr : np.concatenate((arr[:, 1:], arr[:, [-1]]), axis=1)
# p2 = lambda arr : np.concatenate((arr[1:, :], arr[[-1], :]), axis=0)
# p3 = lambda arr : np.concatenate((arr[:, [0]], arr[:, 0:-1]), axis=1)
# p4 = lambda arr : np.concatenate((arr[[0], :], arr[0:-1, :]), axis=0)

# g = arr
# t = g[:, :, [1]]
# lt, at, bt = cv2.split(g)
# tt = np.concatenate((g[:, 1:, :], g[:, [-1], :]), axis=1)
# print(tt)
# last_pad_1 = lambda arr: np.concatenate((arr[:, 1:], arr[:, -1:]), axis=1)
# ttt = last_pad_1(g)
# gx1 = np.concatenate((g[:, :, 1:], g[:, :, [-1]]), axis=2)
# gx2 = np.concatenate((g[:, :, [0]], g[:, :, 0:-1]), axis=2)
# gy1 = np.concatenate((g[:, 1:, :], g[:, [-1], :]), axis=1)
# gy2 = np.concatenate((g[:, [0], :], g[:, 0:-1, :]), axis=1)
# gx = np.subtract(gx1, gx2)
# gy = np.subtract(gy1, gy2)
# dI = np.sqrt(np.sum((np.add(gx ** 2, gy ** 2)), axis=0))

# h = 2 ** (-level)
# psi = 256 * dI / 5
# psi[psi > 1] = 1
# phi = 30. / (1 + 10 * dI / max(smoothness, eps)) * h
#
# gx1 = np.concatenate((arr[:, :, 1:], arr[:, :, [-1]]), axis=2)
# gx2 = np.concatenate((arr[:, :, [0]], arr[:, :, 0:-1]), axis=2)
# gy1 = np.concatenate((arr[:, 1:, :], arr[:, [-1], :]), axis=1)
# gy2 = np.concatenate((arr[:, [0], :], arr[:, 0:-1, :]), axis=1)
# phi1 = np.concatenate((arr1[:, 1:], arr1[:, [-1]]), axis=1)
# phi2 = np.concatenate((arr1[1:, :], arr1[[-1], :]), axis=0)
# phi3 = np.concatenate((arr1[:, [0]], arr1[:, 0:-1]), axis=1)
# phi4 = np.concatenate((arr1[[0], :], arr1[0:-1, :]), axis=0)
# print(phi4)
#
# psi = 256*arr/5
# psi[psi > 1] = 1
# print(psi)
# print(np.concatenate((arr[:, :, 1:], arr[:, :, :-1]), axis=1))