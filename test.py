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

# source_path = "images/autumn.jpg"
# target_path = "images/fallingwater.jpg"
# source_path = "images/ocean_day.jpg"
# target_path = "images/ocean_sunset.jpg"
# source_path = "images/scotland_house.png"
# target_path = "images/scotland_plain.png"
# transfer_path = "images/failed.png"
#
#
# # target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
# # # print(f'{target_rgb.min()}, {target_rgb.max()}')
# # # print(target_rgb[:1])
# # result = cv2.imread(result_path, cv2.IMREAD_COLOR)
# # result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
# # print(f'{result_rgb.min()}, {result_rgb.max()}')
# # print(result_rgb[:1])
# # transfer = color_transfer_orginal(source, target, clip=True, preserve_paper=True)
# source_bgr = cv2.imread(source_path, cv2.IMREAD_COLOR)
# source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_RGB2BGR)
# # test1 = cx_rgb2lab_nestedloop(source_rgb, True)
# # print(test1[:1])
# # test2 = cx_rgb2lab_numpy(source_rgb, True)
# # print(test2[:1])
# # test3 = cx_lab2rgb_nestedloop(test1, True)
# # print(test3[:1])
# # test4 = cx_lab2rgb_numpy(test2, True)
# # print(test4[:1])
# transfer = ColorXfer(source_path, target_path, model='opencv')
# cv2.imwrite(transfer_path, transfer)

def gram_schmidtA(A):
    eps = np.finfo(np.float).eps
    Q = np.array(A, dtype='float32')# Make B as a copy of A, since we're going to alter it's values.
    # Loop over all vectors, starting with zero, label them with i
    for i in range(Q.shape[1]):
        # Inside that loop, loop over all previous vectors, j, to subtract.
        for j in range(i):
            # Complete the code to subtract the overlap with previous vectors.
            # you'll need the current vector B[:, i] and a previous vector B[:, j]
            Q[:, i] = Q[:, i] - Q[:, i] @ Q[:, j] * Q[:, j]
        # Next insert code to do the normalisation test for B[:, i]
        if np.linalg.norm(Q[:, i]) > eps:
            Q[:, i] = Q[:, i] / np.linalg.norm(Q[:, i])
        else:
            Q[:, i] = np.zeros_like(Q[:, i])
    # Finally, we return the result:
    R = Q * A
    return Q, R

def gram_schmidt1(A):
    eps = np.finfo(np.float).eps
    m, n = A.shape
    ap = A.astype(np.float)
    Q = np.zeros((m, n)).astype(np.float)
    R = np.zeros((n, n)).astype(np.float)

    for j in range(n):
        for k in range(j):
            ap[:, j] = ap[:, j] - ap[:, j] @ ap[:, k] * ap[:, k]
        if np.linalg.norm(ap[:, j]) > eps:
            Q[:, j] = ap[:, j] / np.linalg.norm(ap[:, j])
        else:
            Q[:, j] = np.zeros_like(ap[:, j])
    R = Q @ A
    return Q, R

def gram_schmidt2(A):
    """Orthogonalize a set of vectors stored as the columns of matrix A."""
    # Get the number of vectors.
    n = A.shape[1]
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        A[:, j] = A[:, j] / np.linalg.norm(A[:, j])


    return A

def gram_schmidt3(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for k in range(j):
            q = Q[:, k]
            R[k, j] = q.dot(v)
            v = v - R[k, j] * q

    norm = np.linalg.norm(v)
    Q[:, j] = v / norm
    R[j, j] = norm

    return Q, R

def gram_schmidt(A):
    """Orthogonalize a set of vectors stored as the columns of matrix A."""
    # Get the number of vectors.
    eps = np.finfo(np.float).eps
    asave = np.copy(A.astype(np.float))
    n = A.shape[1]
    Q = np.zeros_like(A).astype(np.float)
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            mult = (np.dot(A[:, j], A[:, k])) / (np.dot(A[:, k], A[:, k]))
            A[:, j] = A[:, j] - np.dot(mult, A[:, k])

    for j in range(n):
        if np.linalg.norm(A[:, j]) >= np.sqrt(eps):
            Q[:, j] = A[:, j] / np.linalg.norm(A[:, j])
        else:
            print('Columns of A are linearly dependent.')

    return Q, np.dot(Q.T, asave)

def hyperspherical2cartesianT(x):
    c = np.zeros(len(x) + 1)
    sk = 1

    for k in range(len(x)):
        tt = x[k]
        t = sk * np.cos(tt)
        c[k] = t
        sk = sk * np.sin(tt)

    c[len(x)] = sk
    return c

def generate_rotations(ndim, nb_rotations):
    l = []
    if ndim == 2:
        l.append([0, np.pi/2])
    elif ndim == 3:
        l.append([0, 0, np.pi / 2, 0, np.pi / 2, np.pi / 2])
    #else:
        #put here initialisation for higher orders
    l = np.asarray(l)

    print('rotation ')
    for i in range(nb_rotations):
        print(i)
        t = find_next(l, ndim)
        l = np.append(l, [t], 0)

    m = ndim
    rows = l.shape[0]
    b_prev = np.zeros((m, m))
    rotations = np.zeros((rows, m, m))
    for row in range(rows):
        for j in range(m):
            x = []
            for k in range(ndim - 1):
                c = k + (j * (ndim - 1))
                x.append(l[row, c])
            h = hyperspherical2cartesianT(x)
            b_prev[j, :] = h
        t = b_prev.T
        q, r = gram_schmidtA(t)
        b_prev = q.T
        rotations[row, ::] = b_prev

def find_next(l, ndim):
    prevx = l
    nprevx = prevx.shape[0]
    hdim = ndim - 1
    m = ndim

    b_prev = np.zeros((m, m))
    c_prevx = np.zeros((nprevx * m, ndim))
    for i in range(nprevx):
        for j in range(m):
            y = []
            for k in range(hdim):
                c = k + (j * hdim)
                y.append(prevx[i, c])
            h = hyperspherical2cartesianT(y)
            b_prev[j, :] = h
        t = b_prev.T
        q, r = gram_schmidtA(t)
        b_prev = q.T
        c_prevx[i, :] = b_prev

    def myfun(x1):
        """
        https://www.youtube.com/watch?v=cXHvC_FGx24 find the solution
        :param x:
        :param m:
        :param ndim:
        :param c_prevx:
        :return:
        """
        c_x = np.zeros(m, ndim)

        for i in range(m):
            y = []
            for k in range(hdim):
                c = k + (i * hdim)
                y.append(x1[i, c])
            h = hyperspherical2cartesianT(y)
            c_x[i, :] = h
        t = c_x.T
        q, r = gram_schmidtA(t)
        c_x = q.T
        f = 0

        for i in range(m):
            for p in range(c_prevx.shape[0]):
                d = (c_prevx[p, :] - c_x[i, :]) * (c_prevx[p, :] - c_x[i, :])
                f = f + 1 / (1 + d.T)
                d = (c_prevx[p, :] + c_x[i, :]) * (c_prevx[p, :] + c_x[i, :])
                f = f + 1 / (1 + d.T)

        return f

    for i in range(10):
        x0 = random.randint(1, hdim * m) * np.pi - np.pi / 2
        x = scipy.optimize.minimize(myfun, x0, options={'disp':True})
        f = myfun(x)
        if f < minf:
            minf = f
            mix = x

    return x

nb_rotations = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]]).astype(np.float)


x = generate_rotations(3, 10)
# Q1.shape, R1.shape
# test = np.abs(A - Q1.dot(R1).sum()) < 1e-6

# print("A:")
# pprint(A)
# print("Q:")
# pprint(Q*-1)
# # pprint(test)
# print("Q1:")
# pprint(Q1)
#
# print("R:")
# pprint(R*-1)
print("R1:")
# pprint(R1)

