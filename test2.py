import numpy as np

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

arr = np.array([[[0,  1,  2, 3],
                 [4,  5, 6, 7],
                 [8, 9, 10, 11]],
                [[12, 13, 14, 15],
                 [16, 17, 18, 19],
                 [20, 21, 22, 23]]])
[k, vres, hres] = arr.shape
print('k=' + str(k) + ' vres=' + str(vres) + ' hres=' + str(hres))
print(arr[:, :, 1:hres, [2,]])
#print(np.concatenate((arr[:, :, 1:], arr[:, :, :-1]), axis=1))