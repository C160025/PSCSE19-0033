import numpy as np
import random
import scipy.optimize

def hyperspherical2cartesianT(x):
    """
    must recode and reduce computation time
    :param x:
    :return:
    """
    c = np.zeros(len(x) + 1)
    sk = 1

    for k in range(len(x)):
        c[k] = sk * np.cos(x[k])
        sk = sk * np.sin(x[k])
        # tt = x[k]
        # t = sk * np.cos(tt)
        # c[k] = t
        # sk = sk * np.sin(tt)

    c[len(x)] = sk
    return c

def gram_schmidt(a):
    """
     must recode and reduce computation time *******
    Gram-Schmidt orthogonalization of the columns of a.
    The columns of A are assumed to be linearly independent.
    :param a: numpy array with astype np.float
    :return: q and r in numpy array with astype np.float
    """
    # Q, R = np.linalg.qr(A) Q*-1, R*-1
    # Get the number of vectors.
    eps = np.finfo(np.float).eps
    asave = np.copy(a.astype(np.float))
    n = a.shape[1]
    q = np.zeros_like(a).astype(np.float)
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            mult = (np.dot(a[:, j], a[:, k])) / (np.dot(a[:, k], a[:, k]))
            a[:, j] = a[:, j] - np.dot(mult, a[:, k])

    for j in range(n):
        if np.linalg.norm(a[:, j]) >= np.sqrt(eps):
            q[:, j] = a[:, j] / np.linalg.norm(a[:, j])
        else:
            print('Columns of A are linearly dependent.')

    r = np.dot(q.T, asave)
    return q, r

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
        #t = find_next(l, ndim)
        #l = np.append(l, [t], 0)
        l = np.append(l, [find_next(l, ndim)], 0)

    m = ndim
    rows = l.shape[0]
    b_prev = np.zeros((m, m))
    rotations = np.zeros((rows, m, m))
    for row in range(rows):
        for j in range(m):
            x = []
            for k in range(ndim - 1):
                x.append(l[row, k + (j * (ndim - 1))])
            b_prev[j, :] = hyperspherical2cartesianT(x)
        #         c = k + (j * (ndim - 1))
        #         x.append(l[row, c])
        #     h = hyperspherical2cartesianT(x)
        #     b_prev[j, :] = h
        # t = b_prev.T
        q, r = gram_schmidt(b_prev.T)
        #b_prev = q.T
        rotations[row, ::] = q.T

    return rotations

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
                #c = k + (j * hdim)
                #y.append(prevx[i, c])
                y.append(prevx[i, k + (j * hdim)])
            #h = hyperspherical2cartesianT(y)
            #b_prev[j, :] = h
            b_prev[j, :] = hyperspherical2cartesianT(y)
        # t = b_prev.T
        q, r = gram_schmidt(b_prev.T)
        #b_prev = q.T
        c_prevx = q.T #b_prev

    c_prevx = np.asarray(c_prevx)

    minf = 1.e1000
    for i in range(10):
        x0 = np.random.random((hdim * m,)) * np.pi - np.pi / 2
        # x = scipy.optimize.minimize(myfun, x0, options={'disp':True})
        x = scipy.optimize.fmin(myfun, x0, args=(m, ndim, c_prevx), xtol=1e-10)
        f = myfun(x, m, ndim, c_prevx)
        if f < minf:
            minf = f
            mix = x

    return x

def myfun(x1, m, ndim, c_prevx):
    """
    https://www.youtube.com/watch?v=cXHvC_FGx24 find the solution
    :return:
    """
    c_x = np.zeros((m, ndim))
    hdim = ndim - 1
    for i in range(m):
        y = []
        for k in range(hdim):
            c = k + (i * hdim)
            y.append(x1[c])
        h = hyperspherical2cartesianT(y)
        c_x[i, :] = h
    t = c_x.T
    q, r = gram_schmidt(t)
    c_x = q.T
    f = 0

    for i in range(m):
        for p in range(c_prevx.shape[0]):
            d = np.dot((c_prevx[p, :] - c_x[i, :]), (c_prevx[p, :] - c_x[i, :]).T)
            f = f + 1 / (1 + d.T)
            d = np.dot((c_prevx[p, :] + c_x[i, :]), (c_prevx[p, :] + c_x[i, :]).T)
            f = f + 1 / (1 + d.T)

    return f
