import math
import numpy as np
import cv2
import time
import scipy as sp
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
        q, r = gram_schmidt(b_prev.T)
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
                y.append(prevx[i, k + (j * hdim)])
            b_prev[j, :] = hyperspherical2cartesianT(y)
        q, r = gram_schmidt(b_prev.T)
        c_prevx = q.T

    c_prevx = np.asarray(c_prevx)

    minf = 1.e1000
    for i in range(10):
        x0 = np.random.random((hdim * m,)) * np.pi - np.pi / 2
        # x = scipy.optimize.minimize(myfun, x0, options={'disp':True})
        x = scipy.optimize.fmin(myfun, x0, args=(m, ndim, c_prevx), xtol=1e-10, disp=False)
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

x = generate_rotations(3, 10)
print(x)

def random_rotations(row=6, col=3):
    """
    generate orthogonal matrices for pdf transfer. Random rotation.
    :return:
    """
    assert row > 0
    rotation_matrices = [np.eye(col)]
    rotation_matrices.extend([np.matmul(rotation_matrices[0], sp.rvs(dim=col)) for _ in range(row - 1)])
    return rotation_matrices

#  F. Pitié 2007 Automated colour grading using colour distribution transfer
def optimal_rotations():
    """
    generate orthogonal matrices for pdf transfer. Optimal rotation.
    :return:
    """
    three_dim_optimised_rotation = [np.array([
        [[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000]],  #1
        [[0.333333, 0.666667, 0.666667], [0.666667, 0.333333, -0.666667], [-0.666667, 0.666667, -0.333333]], #2
        [[0.577350, 0.211297, 0.788682], [-0.577350, 0.788668, 0.211352], [0.577350, 0.577370, -0.577330]], #3
        [[0.577350, 0.408273, 0.707092], [-0.577350, -0.408224, 0.707121], [0.577350, -0.816497, 0.000029]], #4
        [[0.332572, 0.910758, 0.244778], [-0.910887, 0.242977, 0.333536], [-0.244295, 0.333890, -0.910405]], #5
        [[0.243799, 0.910726, 0.333376], [0.910699, -0.333174, 0.244177], [-0.333450, -0.244075, 0.910625]], #6
        [[-0.109199, 0.810241, 0.575834], [0.645399, 0.498377, -0.578862], [0.756000, -0.308432, 0.577351]], #7
        [[0.759262, 0.649435, -0.041906], [0.143443, -0.104197, 0.984158], [0.634780, -0.753245, -0.172269]], #8
        [[0.862298, 0.503331, -0.055679], [-0.490221, 0.802113, -0.341026], [-0.126988, 0.321361, 0.938404]], #9
        [[0.982488, 0.149181, 0.111631], [0.186103, -0.756525, -0.626926], [-0.009074, 0.636722, -0.771040]], #10
        [[0.687077, -0.577557, -0.440855], [0.592440, 0.796586, -0.120272], [-0.420643, 0.178544, -0.889484]], #11
        [[0.463791, 0.822404, 0.329470], [0.030607, -0.386537, 0.921766], [-0.885416, 0.417422, 0.204444]], #12
    ])]
    return three_dim_optimised_rotation

# Erik Reinhard 2001 Color Transfer between Images
def cx_rgb2lab(image_rgb, log10):
    """
    Color space conversion from RGB to lab space referencing from paper
    Referencing from http://erikreinhard.com/papers/colourtransfer.pdf paper
    :param image_rgb: image in RGB color space on numpy array
    :param log10:
    :return: image in lab color space on numpy array
    """
    t = time.time()
    # equation 2 to convert RGB space to XYZ space
    rgb2xyz_eq = [[0.5141, 0.3239, 0.1604],
                  [0.2651, 0.6702, 0.0641],
                  [0.0241, 0.1228, 0.8444]]

    # equation 3 to convert XYZ space to LMS space
    xyz2lms_eq = [[0.3897, 0.6890, -0.0787],
                  [-0.2298, 1.1834, 0.0464],
                  [0.0000, 0.0000, 1.0000]]

    # dot product of equation 2 and 3 to convert RGB space to LMS space (more precise)
    rgb2lms_eq1 = np.matmul(xyz2lms_eq, rgb2xyz_eq)

    # equation 4 to to convert RGB space to LMS space
    rgb2lms_eq2 = [[0.3811, 0.5783, 0.0402],
                   [0.1967, 0.7244, 0.0782],
                   [0.0241, 0.1288, 0.8444]]

    # left of equation 6 to convert LMS space to lab space
    lms2lab_eq1 = [[1 / math.sqrt(3), 0.0000, 0.0000],
                   [0.0000, 1 / math.sqrt(6), 0.0000],
                   [0.0000, 0.0000, 1 / math.sqrt(2)]]

    # right of equation 6 to convert LMS space to lab space
    lms2lab_eq2 = [[1.0000, 1.0000, 1.0000],
                   [1.0000, 1.0000, -2.0000],
                   [1.0000, -1.0000, 0.0000]]

    # dot product of equation 6 to convert LMS space to lab space
    lms2lab_eq = np.matmul(lms2lab_eq1, lms2lab_eq2)

    # split RGB into individual channel space for color space conversion
    (r, g, b) = cv2.split(image_rgb)

    # convert RGB space to LMS space
    L = r * rgb2lms_eq1[0][0] + g * rgb2lms_eq1[0][1] + b * rgb2lms_eq1[0][2]
    M = r * rgb2lms_eq1[1][0] + g * rgb2lms_eq1[1][1] + b * rgb2lms_eq1[1][2]
    S = r * rgb2lms_eq1[2][0] + g * rgb2lms_eq1[2][1] + b * rgb2lms_eq1[2][2]

    # to prevent divide by zero encountered in log python numpy machine epsilon
    eps = np.finfo(np.float32).eps

    # equation 5 to eliminate the skew
    if (log10):
        L = np.log10(L.clip(min=eps))
        M = np.log10(M.clip(min=eps))
        S = np.log10(S.clip(min=eps))

    # convert LMS space to lab space
    l = L * lms2lab_eq[0][0] + M * lms2lab_eq[0][1] + S * lms2lab_eq[0][2]
    a = L * lms2lab_eq[1][0] + M * lms2lab_eq[1][1] + S * lms2lab_eq[1][2]
    b = L * lms2lab_eq[2][0] + M * lms2lab_eq[2][1] + S * lms2lab_eq[2][2]

    # merge individual channel into lab color space
    image_lab = cv2.merge([l, a, b]).astype(np.float32)

    print("took {} s to be remove after testing phase".format(time.time() - t))
    return image_lab

# Erik Reinhard 2001 Color Transfer between Images
def cx_lab2rgb(image_lab, power10):
    """
    Color space conversion from lab to RGB space.
    Referencing from http://erikreinhard.com/papers/colourtransfer.pdf paper
    :param image_lab: image in lab color space on numpy array
    :param power10:
    :return: image in RGB color space on numpy array
    """
    t = time.time()
    # left of equation 8 to convert lab space to LMS space
    lab2lms_eq1 = [[1.0000, 1.0000, 1.0000],
                   [1.0000, 1.0000, -1.0000],
                   [1.0000, -2.0000, 0.0000]]

    # right of equation 8 to convert lab space to LMS space
    lab2lms_eq2 = [[math.sqrt(3)/3, 0.0000, 0.0000],
                   [0.0000, math.sqrt(6)/6, 0.0000],
                   [0.0000, 0.0000, math.sqrt(2)/2]]

    # dot product of equation 8 to convert lab space to LMS space
    lab2lms_eq = np.matmul(lab2lms_eq1, lab2lms_eq2)

    # equation 9 to convert LMS space to RGB space
    lms2rgb_eq = [[4.4679, -3.5876, 0.1193],
                  [-1.2186, 2.3809, -0.1624],
                  [0.0497, -0.2439, 1.2045]]

    # split lab into individual channel space for color space conversion
    (l, a, b) = cv2.split(image_lab)

    # convert lab space to LMS space
    L = l * lab2lms_eq[0][0] + a * lab2lms_eq[0][1] + b * lab2lms_eq[0][2]
    M = l * lab2lms_eq[1][0] + a * lab2lms_eq[1][1] + b * lab2lms_eq[1][2]
    S = l * lab2lms_eq[2][0] + a * lab2lms_eq[2][1] + b * lab2lms_eq[2][2]

    # raise back to linear space
    if (power10):
        L = np.power(10, L)
        M = np.power(10, M)
        S = np.power(10, S)

    # convert LMS space to RGB space
    r = L * lms2rgb_eq[0][0] + M * lms2rgb_eq[0][1] + S * lms2rgb_eq[0][2]
    g = L * lms2rgb_eq[1][0] + M * lms2rgb_eq[1][1] + S * lms2rgb_eq[1][2]
    b = M * lms2rgb_eq[2][0] + M * lms2rgb_eq[2][1] + S * lms2rgb_eq[2][2]

    # merge individual channel into RGB color space
    img_rgb = cv2.merge([r, g, b]).astype(np.float32)

    print("took {} s to be remove after testing phase".format(time.time() - t))
    return img_rgb