import cv2
import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import special_ortho_group
from scipy import interpolate
from color_transfer.utils import cx_rgb2lab, cx_lab2rgb, optimal_rotations, random_rotations
from color_transfer.generate_rotations import generate_rotations


eps = np.finfo(np.float).eps

# The linear Monge-Kantorovitch linear colour mapping for
# example-based colour transfer. F. Pitié and A. Kokaram (2007) In 4th
# IEE European Conference on Visual Media Production (CVMP'07). London,
# November.

def REGRAIN_CX(source_rgb, target_rgb):
    idt_rgb = idt_cx(source_rgb, target_rgb, bins=300, n_rot=30, relaxation=1)
    return regrain_cx(source_rgb, idt_rgb)

def regrain_cx(source_rgb, target_rgb):#, varargin=None):
    """
    https://github.com/pengbo-learn/python-color-transfer/blob/master/python_color_transfer/color_transfer.py
    :param source_rgb:
    :param target_rgb:
    :param varargin:
    :return:
    """
    source = np.divide(source_rgb, 255)
    target = np.divide(target_rgb, 255)
    # numvarargs = len(varargin)
    # if numvarargs > 1:
    #     print('regrain :TooManyInputs requires at most 1 optional input')
    # optargs = np.ones(1)
    # optargs[1: numvarargs] = varargin
    # [smoothness] = optargs[:]

    source_rgb_regrain = np.zeros_like(source_rgb)
    output = regrain_rec(source_rgb_regrain, source, target, nbits=np.array([4, 16, 32, 64, 64, 64]), smoothness=1, level=0)
    return np.multiply(output, 255).astype(np.uint8)

def solve(source_rgb_regrain, source_rgb, target_rgb, nbits, smoothness, level):
    
    [vres, hres, k] = source_rgb.shape

    g = source_rgb

    gx1 = np.concatenate((g[:, 1:, :], g[:, [-1], :]), axis=1)
    gx2 = np.concatenate((g[:, [0], :], g[:, 0:-1, :]), axis=1)
    gy1 = np.concatenate((g[1:, :, :], g[[-1], :, :]), axis=0)
    gy2 = np.concatenate((g[[0], :, :], g[0:-1, :, :]), axis=0)
    gx = np.subtract(gx1, gx2)
    gy = np.subtract(gy1, gy2)
    dI = np.sqrt(np.sum((np.add(gx**2, gy**2)), axis=2))

    h = 2 ** (-level)
    psi = np.divide(np.multiply(256, dI), 5)
    psi[psi > 1] = 1
    phi = 30. / (1 + 10 * dI / max(smoothness, eps)) * h

    p1 = lambda arr: np.concatenate((arr[:, 1:], arr[:, -1:]), axis=1)
    p2 = lambda arr: np.concatenate((arr[1:, :], arr[-1:, :]), axis=0)
    p3 = lambda arr: np.concatenate((arr[:, :1], arr[:, :-1]), axis=1)
    p4 = lambda arr: np.concatenate((arr[:1, :], arr[:-1, :]), axis=0)

    phi1 = np.divide(np.add(p1(phi), phi), 2)
    phi2 = np.divide(np.add(p2(phi), phi), 2)
    phi3 = np.divide(np.add(p3(phi), phi), 2)
    phi4 = np.divide(np.add(p4(phi), phi), 2)

    rho = 1/5
    for i in range(nbits):
        #den = psi + phi1 + phi2 + phi3 + phi4 #np.add(np.add(np.add(np.add(psi, phi1), phi2), phi3), phi4)
        den = np.add(np.add(np.add(np.add(psi, phi1), phi2), phi3), phi4) + eps
        #xx = np.repeat(psi[:, :, np.newaxis], k, axis=2)
        #xxx = xx * target_rgb
        num = (np.repeat(psi[:, :, np.newaxis], k, axis=2) * target_rgb
               + np.repeat(phi1[:, :, np.newaxis], k, axis=2) * (p1(source_rgb_regrain) - p1(source_rgb) + source_rgb)
               + np.repeat(phi2[:, :, np.newaxis], k, axis=2) * (p2(source_rgb_regrain) - p2(source_rgb) + source_rgb)
               + np.repeat(phi3[:, :, np.newaxis], k, axis=2) * (p3(source_rgb_regrain) - p3(source_rgb) + source_rgb)
               + np.repeat(phi4[:, :, np.newaxis], k, axis=2) * (p4(source_rgb_regrain) - p4(source_rgb) + source_rgb))
        source_rgb_regrain = num / np.repeat(den[:, :, np.newaxis], k, axis=2) * (1 - rho) + rho * source_rgb_regrain

    return source_rgb_regrain

def regrain_rec(source_rgb_regrain, source_rgb, target_rgb, nbits, smoothness, level):
    [vres, hres, k] = source_rgb.shape
    vres2 = int(np.ceil(vres / 2))
    hres2 = int(np.ceil(hres / 2))

    if len(nbits) > 1 and vres2 > 20 and hres2 > 20:
        resize_source_rgb = cv2.resize(source_rgb, (hres2, vres2), interpolation=cv2.INTER_LINEAR)
        resize_target_rgb = cv2.resize(target_rgb, (hres2, vres2), interpolation=cv2.INTER_LINEAR)
        resize_source_rgb_regrain = cv2.resize(source_rgb_regrain, (hres2, vres2), interpolation=cv2.INTER_LINEAR)
        resize_source_rgb_regrain = regrain_rec(resize_source_rgb_regrain, resize_source_rgb, resize_target_rgb, nbits[1:], smoothness, level=level+1)
        source_rgb_regrain = cv2.resize(resize_source_rgb_regrain, (hres, vres), interpolation=cv2.INTER_LINEAR)

    source_rgb_regrain = solve(source_rgb_regrain, source_rgb, target_rgb, nbits[0], smoothness, level)

    return source_rgb_regrain

def MKL_CX(source_rgb, target_rgb, conversion):
    if conversion == 'matlab':
        return matlab_mkl_cx(source_rgb, target_rgb)
    elif conversion == 'recode':
        return mkl_cx(source_rgb, target_rgb)

def mkl_cx(source_rgb, target_rgb):
    """
    Probability Density Function (PDF) Monge-Kantorovitch linear (MKL) Color transfer (CX)
    :param source_rgb: source image in RGB color space (0-255) on numpy array
    :param target_rgb: target image in RGB color space (0-255) on numpy array
    :return: output_rgb: corrected image in RGB color space (0-255) on numpy array
    """
    source = np.divide(source_rgb, 255)
    target = np.divide(target_rgb, 255)

    #source_rgb = pd.DataFrame(source.reshape(-1, source.shape[-1]), columns=['r', 'g', 'b']).values
    #target_rgb = pd.DataFrame(target.reshape(-1, target.shape[-1]), columns=['r', 'g', 'b']).values
    source_rgb = source.reshape(-1, source.shape[-1])
    target_rgb = target.reshape(-1, target.shape[-1])

    a = np.cov(source_rgb.T)
    b = np.cov(target_rgb.T)

    Da2, Ua = np.linalg.eig(a)
    Da = np.diag(np.sqrt(Da2.clip(eps, None)))

    C = np.dot(np.dot(np.dot(np.dot(Da, Ua.T), b), Ua), Da)

    Dc2, Uc = np.linalg.eig(C)
    Dc = np.diag(np.sqrt(Dc2.clip(eps, None)))

    Da_inv = np.diag(1. / (np.diag(Da)))

    t = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(Ua, Da_inv), Uc), Dc), Uc.T), Da_inv), Ua.T)

    mx0 = np.mean(source_rgb, axis=0)
    mx1 = np.mean(target_rgb, axis=0)

    xr = np.add(np.dot(np.subtract(source_rgb, mx0), t), mx1)

    a_result = xr
    xx = a_result.reshape(source.shape)
    output_rgb = np.multiply(xx, 255).astype(np.uint8)

    return output_rgb

def matlab_mkl_cx(source_rgb, target_rgb):
    """

    :param source_rgb: source image in RGB color space (0-255) on numpy array
    :param target_rgb: target image in RGB color space (0-255) on numpy array
    :return: output_rgb: corrected image in RGB color space (0-255) on numpy array
    """
    source = np.divide(source_rgb, 255)
    target = np.divide(target_rgb, 255)

    if source.ndim != 3:
        print('pictures must have 3 dimensions')
    # reshape images
    source_rgb = source.reshape(-1, source.shape[-1])
    target_rgb = target.reshape(-1, target.shape[-1])

    a = np.cov(source_rgb.T)
    b = np.cov(target_rgb.T)

    t = mkl(a, b)

    mx0 = np.mean(source_rgb, axis=0)
    mx1 = np.mean(target_rgb, axis=0)

    xr = np.add(np.dot(np.subtract(source_rgb, mx0), t), mx1)

    irv = xr.values

    xx = irv.reshape(source.shape)
    output_rgb = np.multiply(xx, 255).astype(np.uint8)

    return output_rgb

def mkl(a, b):
    da2, ua = np.linalg.eig(a)
    da = np.diag(np.sqrt(da2.clip(eps, None)))
    c = np.dot(np.dot(np.dot(np.dot(da, ua.T), b), ua), da)
    dc2, uc = np.linalg.eig(c)
    dc = np.diag(np.sqrt(dc2.clip(eps, None)))
    da_inv = np.diag(1. / (np.diag(da)))
    t = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(ua, da_inv), uc), dc), uc.T), da_inv), ua.T)
    return t
#
def IDT_CX(source_rgb, target_rgb, conversion):
    """
    Probability Density Function (PDF) Iterative Distribution Transfer (IDT) Color transfer (CX)
    :param source_rgb:
    :param target_rgb:
    :param conversion:
    :return:
    """
    if conversion == 'matlab':
        #return matlab2_idt_cx(source_rgb, target_rgb, bins=300, nb_iterations=10, rotation_type='')
        return matlab_idt_cx(source_rgb, target_rgb, nb_iterations=30, rotation_type='')
    elif conversion == 'recode':
        return idt_cx(source_rgb, target_rgb, bins=300, n_rot=30, relaxation=1)

# F. Pitie 2005 N-Dimensional PDF Transfer Iterative Distribution Transfer
def idt_cx(source_rgb, target_rgb, bins, n_rot, relaxation=1):
    """
    Probability Density Function (PDF) Iterative Distribution Transfer (IDT) Color transfer (CX)
    :param source_rgb:
    :param target_rgb:
    :param bins:
    :param n_rot:
    :param relaxation:
    :return:
    """

    eps32 = 1e-6
    source = np.divide(source_rgb, 255)
    target = np.divide(target_rgb, 255)
    if source.ndim != 3:
        print('pictures must have 3 dimensions')
        # reshape images
    source = source.transpose().reshape(source.shape[-1], -1)
    target = target.transpose().reshape(target.shape[-1], -1)

    n_dims = source.shape[0]
    d0 = source
    d1 = target
    rot = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2 / 3, 2 / 3, -1 / 3], [2 / 3, -1 / 3, 2 / 3], [-1 / 3, 2 / 3, 2 / 3]])
    rotation = np.zeros((n_rot, rot.shape[0], rot.shape[1]))
    for i in range(n_rot):
        #  create a random orthonormal matrix
        ortho = special_ortho_group.rvs(n_dims).astype(np.float)
        r = rot @ ortho if i > 0 else rot
        d0r = np.matmul(r, d0)
        d1r = np.matmul(r, d1)
        d_r = np.empty_like(d0r)
        nb_projs = r.shape[0]
        for j in range(nb_projs):
            lo = min(d0r[j].min(), d1r[j].min()) - eps
            hi = max(d0r[j].max(), d1r[j].max()) + eps
            p0r, edges = np.histogram(d0r[j], bins=bins-1, range=[lo, hi])
            p1r, _ = np.histogram(d1r[j], bins=bins-1, range=[lo, hi])
            #xs = xs[1:]
            #cp0r = np.add(p0r, eps).cumsum().astype(np.float)
            cp0r = np.add(p0r, eps).cumsum()
            d_0 = cp0r / cp0r[-1]
            #cp0r /= cp0r[-1]
            #cp1r = np.add(p1r, eps).cumsum().astype(np.float)
            cp1r = np.add(p1r, eps).cumsum()
            d_1 = cp1r / cp1r[-1]
            #cp1r /= cp1r[-1]
            f = np.interp(d_0, d_1, edges[:-1])
            d_r[j] = np.interp(d0r[j], edges[1:], f.T, left=0, right=bins-1)

        d0 = relaxation * np.matmul(np.linalg.pinv(r), (d_r - d0r)) + d0
    output_r = np.multiply(d0[0, :].reshape((source_rgb.shape[1], source_rgb.shape[0])).T, 255).astype(np.uint8)
    output_g = np.multiply(d0[1, :].reshape((source_rgb.shape[1], source_rgb.shape[0])).T, 255).astype(np.uint8)
    output_b = np.multiply(d0[2, :].reshape((source_rgb.shape[1], source_rgb.shape[0])).T, 255).astype(np.uint8)
    return cv2.merge([output_r, output_g, output_b])

def matlab2_idt_cx(source_rgb, target_rgb, bins, nb_iterations, rotation_type):
    if (source_rgb.ndim != 3):
        print('pictures must have 3 dimensions')
        # Normalized RGB
    source = np.divide(source_rgb, 255)
    (source_l, source_a, source_b) = cv2.split(source)
    target = np.divide(target_rgb, 255)
    (target_l, target_a, target_b) = cv2.split(target)
    nb_channels = source.ndim
    # reshape images as 3xN matrices
    # source_rgb_reshape = []
    # target_rgb_reshape = []
    # for i in range(nb_channels):
    #     source_rgb_reshape[i, :] = np.reshape(source_rgb[:, :, i], (1, source_rgb.shape[0] * source_rgb.shape[1]))
    #     target_rgb_reshape[i, :] = np.reshape(target_rgb[:, :, i], (1, target_rgb.shape[0] * target_rgb.shape[1]))
    source_rgb_reshape = source.transpose().reshape(source.shape[-1], -1)  # source.reshape(-1, source.shape[-1])
    target_rgb_reshape = target.transpose().reshape(target.shape[-1], -1)  # target.reshape(-1, target.shape[-1])
    # building a sequence of (almost) random projections
    rot = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2 / 3, 2 / 3, -1 / 3], [2 / 3, -1 / 3, 2 / 3], [-1 / 3, 2 / 3, 2 / 3]])
    rotation = np.zeros((nb_iterations, rot.shape[0], rot.shape[1]))
    if rotation_type == 'optimal':
        rotation = optimal_rotations
    elif rotation_type == 'random':
        rotation[0, :] = rot
        for i in range(1, nb_iterations):
            r = random_rotations()
            rotation[i, :] = r
    elif rotation_type == 'generate':
        rotation = generate_rotations(nb_channels, nb_iterations)
    else:
        rotation[0, :] = rot
        for i in range(1, nb_iterations):
            r = special_ortho_group.rvs(3).astype(np.float)
            rotation[i, :, ] = rot.dot(r)
    #data_rotation = pdf_transfer_matlab(source_rgb_reshape, target_rgb_reshape, rotation, [1])
    numvarargs = len([1])
    if numvarargs > 1:
        print('pdf_transfer:TooManyInputs, requires at most 1 optional input')

    # cell array = numpy object array
    optargs = np.ones(1)
    optargs[1: numvarargs] = 1
    [relaxation] = optargs[:]

    prompt = ''
    for it in range(nb_iterations):
        print('IDT iteration ', it, '/', nb_iterations)
        r = rotation[it]
        nb_projs = r.shape[0]

        # apply rotation
        source_rgb_rot = np.matmul(r, source_rgb_reshape)
        target_rgb_rot = np.matmul(r, target_rgb_reshape)
        source_rgb_rot_ = np.zeros_like(source_rgb_rot)

        # get the marginals, match them, and apply transformation
        for i in range(nb_projs):
            # get the data range
            datamin = min(min(source_rgb_rot[i, :]), min(target_rgb_rot[i, :])) - eps
            datamax = max(max(source_rgb_rot[i, :]), max(target_rgb_rot[i, :])) + eps
            # bin 300 from datamax datamin used numpy linespace #####
            u = np.linspace(datamin, datamax, num=bins + 1)
            # xs = np.array([datamin + (datamax - datamin) * xi / n for xi in range(n + 1)])
            # np.array([j * (datamax - datamin) + datamin for j in range(n)]) sda
            # get the projections
            source_hist_projs, edges = np.histogram(source_rgb_rot[i, :], bins=bins, range=[datamin, datamax])
            target_hist_projs, _ = np.histogram(target_rgb_rot[i, :], bins=bins, range=[datamin, datamax])
            # get the transport map
            #f = one_dim_pdf_transfer_matlab(source_hist_projs, target_hist_projs)
            eps1 = 1e-6
            source_rgb_cumsum = np.cumsum(np.add(source_hist_projs, eps1))
            source_rgb_cs = np.divide(source_rgb_cumsum, source_rgb_cumsum[-1])

            target_rgb_cumsum = np.cumsum(np.add(target_hist_projs, eps1))
            target_rgb_cs = np.divide(target_rgb_cumsum, target_rgb_cumsum[-1])
            #  inversion
            xs = np.linspace(0, bins - 1, num=bins)

            f = np.interp(source_rgb_cs, target_rgb_cs, edges[1:])
            # f = interpolate.interp1d(interpolate.interp1d, target_rgb_cumsum, xs)
            f[source_rgb_cs <= target_rgb_cs[0]] = 0
            f[source_rgb_cs >= target_rgb_cs[-1]] = bins - 1
            xs = u[:-1]
            # apply the mapping
            t1 = np.interp(source_rgb_rot[i, :], xs, f.T)
            t2 = np.subtract(np.interp(source_rgb_rot[i, :], xs, f.T), 1)
            t3 = np.subtract(np.interp(source_rgb_rot[i, :], xs, f.T), 1) / (bins - 1)
            t4 = np.subtract(datamax, datamin)
            t5 = np.multiply(np.subtract(np.interp(source_rgb_rot[i, :], xs, f.T), 1) / (bins - 1), np.subtract(datamax, datamin))
            t6 = np.add(np.multiply(np.subtract(np.interp(source_rgb_rot[i, :], xs, f.T), 1) / (bins - 1), np.subtract(datamax, datamin)), datamin)
            # source_rgb_rot_[i, :] = t6
            source_rgb_rot_[i] = np.interp(source_rgb_rot[i], edges[1:], f, left=0, right=bins)
            # source_rgb_rot_[i, :] = np.add(
            #     np.multiply(np.subtract(np.interp(source_rgb_rot[i, :], xs, f.T), 1) / (bins - 1),
            #                 np.subtract(datamax, datamin)), datamin)

        xx = np.subtract(source_rgb_rot_, source_rgb_rot)
        ri = np.linalg.pinv(r)
        # #(6,3)/(6,173641) cannot divide need to find out solution
        xxx = np.matmul(ri, xx)
        xxxx = relaxation * xxx
        xxxxx = xxxx + source_rgb_reshape
        source_rgb_reshape = relaxation * np.matmul(np.linalg.pinv(r), (source_rgb_rot_ - source_rgb_rot)) + source_rgb_reshape
        #source_rgb_reshape = xxxxx #np.dot(relaxation, (np.matmul(np.linalg.pinv(r), (np.subtract(source_rgb_rot_, source_rgb_rot))))) + source_rgb_reshape

    output_rgb = source
    t1 = source_rgb_reshape.transpose()
    t2 = t1.reshape(source.shape)
    # for i in range(nb_channels):
    #    output_rgb[:, :, i] = np.reshape(data_rotation[i, :], (output_rgb.shape[0], output_rgb.shape[1]))
    xx = np.multiply(t2, 255).astype(np.uint8)

    return xx
def matlab_idt_cx(source_rgb, target_rgb, nb_iterations, rotation_type):
    if(source_rgb.ndim != 3):
        print('pictures must have 3 dimensions')
    # Normalized RGB
    source = np.divide(source_rgb, 255)
    (source_l, source_a, source_b) = cv2.split(source)
    target = np.divide(target_rgb, 255)
    (target_l, target_a, target_b) = cv2.split(target)
    nb_channels = source.ndim
    # reshape images as 3xN matrices
    # source_rgb_reshape = []
    # target_rgb_reshape = []
    # for i in range(nb_channels):
    #     source_rgb_reshape[i, :] = np.reshape(source_rgb[:, :, i], (1, source_rgb.shape[0] * source_rgb.shape[1]))
    #     target_rgb_reshape[i, :] = np.reshape(target_rgb[:, :, i], (1, target_rgb.shape[0] * target_rgb.shape[1]))
    source_rgb_reshape = source.transpose().reshape(source.shape[-1], -1) #source.reshape(-1, source.shape[-1])
    target_rgb_reshape = target.transpose().reshape(target.shape[-1], -1) #target.reshape(-1, target.shape[-1])
    # building a sequence of (almost) random projections
    rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [2 / 3, 2 / 3, -1 / 3], [2 / 3, -1 / 3, 2 / 3], [-1 / 3, 2 / 3, 2 / 3]])
    rotation = np.zeros((nb_iterations, rot.shape[0], rot.shape[1]))
    if rotation_type == 'optimal':
        rotation = optimal_rotations
    elif rotation_type == 'random':
        rotation[0, :] = rot
        for i in range(1, nb_iterations):
            r = random_rotations()
            rotation[i, :] = r
    elif rotation_type == 'generate':
        rotation = generate_rotations(nb_channels, nb_iterations)
    else:
        rotation[0, :] = rot
        for i in range(1, nb_iterations):
            r = special_ortho_group.rvs(3).astype(np.float)
            rotation[i, :, ] = rot.dot(r)
    data_rotation = pdf_transfer_matlab(source_rgb_reshape, target_rgb_reshape, rotation, [1])
    # output_rgb = source
    # t1 = data_rotation.transpose()
    # t2 = t1.reshape(source.shape)
    # #for i in range(nb_channels):
    # #    output_rgb[:, :, i] = np.reshape(data_rotation[i, :], (output_rgb.shape[0], output_rgb.shape[1]))
    # xx = np.multiply(t2, 255).astype(np.uint8)
    output_r = np.multiply(data_rotation[0, :].reshape((source_rgb.shape[1], source_rgb.shape[0])).T, 255).astype(np.uint8)
    output_g = np.multiply(data_rotation[1, :].reshape((source_rgb.shape[1], source_rgb.shape[0])).T, 255).astype(np.uint8)
    output_b = np.multiply(data_rotation[2, :].reshape((source_rgb.shape[1], source_rgb.shape[0])).T, 255).astype(np.uint8)
    return cv2.merge([output_r, output_g, output_b])

def pdf_transfer(source_rgb, target_rgb, optimal=True, n=300, step_size=1):
    """
    :param source_rgb:
    :param target_rgb:
    :param optimal: True = Optimal rotation matrices from paper, False = Random rotation matrices.
    :param n:
    :param step_size:
    :return:
    """
    # reshape (h, w, c) to (c, h*w) wrong here *******************
    [w, c] = source_rgb.shape
    reshape_arr_in = source_rgb.reshape(-1, c).transpose() / 255.
    reshape_arr_ref = target_rgb.reshape(-1, c).transpose() / 255.
    # pdf transfer
    # n times of 1d-pdf-transfer
    arr_out = np.array(reshape_arr_in)
    rotation_matrices = random_rotations
    if optimal:
        rotation_matrices = optimal_rotations
    for rotation_matrix in rotation_matrices:
        rot_arr_in = np.matmul(rotation_matrix, arr_out)
        rot_arr_ref = np.matmul(rotation_matrix, reshape_arr_ref)
        rot_arr_out = np.zeros(rot_arr_in.shape)
        for i in range(rot_arr_out.shape[0]):
            rot_arr_out[i] = one_dim_pdf_transfer(rot_arr_in[i], rot_arr_ref[i], n)
        # func = lambda x, n : self._pdf_transfer_1d(x[:n], x[n:])
        # rot_arr = np.concatenate((rot_arr_in, rot_arr_ref), axis=1)
        # rot_arr_out = np.apply_along_axis(func, 1, rot_arr, rot_arr_in.shape[1])
        rot_delta_arr = rot_arr_out - rot_arr_in
        delta_arr = np.matmul(rotation_matrix.transpose(), rot_delta_arr)
        # np.linalg.solve(rotation_matrix, rot_delta_arr)
        arr_out = step_size * delta_arr + arr_out
    # reshape (c, h*w) to (h, w, c)
    arr_out[arr_out < 0] = 0
    arr_out[arr_out > 1] = 1
    reshape_arr_out = (255. * arr_out).astype('uint8')
    img_arr_out = reshape_arr_out.transpose().reshape(w, c)
    return img_arr_out

def pdf_transfer_matlab(source_rgb, target_rgb, rotations, varargin, n=300):
    """
    :param source_rgb:
    :param target_rgb:
    :param Rotations:
    :param varargin: 1x1 cell array
    :return:
    """
    nb_iterations = len(rotations)

    numvarargs = len(varargin)
    if numvarargs > 1:
        print('pdf_transfer:TooManyInputs, requires at most 1 optional input')

    # cell array = numpy object array
    optargs = np.ones(1)
    optargs[1: numvarargs] = varargin
    [relaxation] = optargs[:]

    prompt = ''
    for it in range(nb_iterations):
        print('IDT iteration ', it, '/', nb_iterations)
        r = rotations[it]
        nb_projs = r.shape[0]

        # apply rotation
        source_rgb_rot = np.matmul(r, source_rgb)
        target_rgb_rot = np.matmul(r, target_rgb)
        source_rgb_rot_ = np.zeros_like(source_rgb_rot)

        # get the marginals, match them, and apply transformation
        for i in range(nb_projs):
            # get the data range
            datamin = min(min(source_rgb_rot[i, :]), min(target_rgb_rot[i, :])) - eps
            datamax = max(max(source_rgb_rot[i, :]), max(target_rgb_rot[i, :])) + eps
            # bin 300 from datamax datamin used numpy linespace #####
            u = np.linspace(datamin, datamax, num=n+1)
            #xs = np.array([datamin + (datamax - datamin) * xi / n for xi in range(n + 1)])
            #np.array([j * (datamax - datamin) + datamin for j in range(n)]) sda
            # get the projections
            source_hist_projs, _ = np.histogram(source_rgb_rot[i, :], u)
            target_hist_projs, _ = np.histogram(target_rgb_rot[i, :], u)
            # get the transport map
            f = one_dim_pdf_transfer_matlab(source_hist_projs, target_hist_projs)
            xs = u[:-1]
            # apply the mapping
            # t1 = np.interp(source_rgb_rot[i, :], xs, f.T)
            # t2 = np.subtract(np.interp(source_rgb_rot[i, :], xs, f.T), 1)
            # t3 = np.subtract(np.interp(source_rgb_rot[i, :], xs, f.T), 1) / (n - 1)
            # t4 = np.subtract(datamax, datamin)
            # t5 = np.multiply(np.subtract(np.interp(source_rgb_rot[i, :], xs, f.T), 1) / (n - 1), np.subtract(datamax, datamin))
            # t6 = np.add(np.multiply(np.subtract(np.interp(source_rgb_rot[i, :], xs, f.T), 1) / (n - 1), np.subtract(datamax, datamin)), datamin)
            # source_rgb_rot_[i, :] = t6
            source_rgb_rot_[i, :] = np.add(np.multiply(np.subtract(np.interp(source_rgb_rot[i, :], xs, f.T), 1) / (n - 1), np.subtract(datamax, datamin)), datamin)

        # xx = np.subtract(source_rgb_rot_, source_rgb_rot)
        # ri = np.linalg.pinv(r)
        # #(6,3)/(6,173641) cannot divide need to find out solution
        # xxx = np.matmul(ri, xx)
        # xxxx = np.dot(relaxation, xxx)
        # xxxxx = np.add(xxxx, source_rgb)
        source_rgb = np.add(np.dot(relaxation, (np.matmul(np.linalg.pinv(r), np.subtract(source_rgb_rot_, source_rgb_rot)))), source_rgb)

    return source_rgb

# F. Pitie 2005 N-Dimensional PDF Transfer
def one_dim_pdf_transfer(source_rgb, target_rgb, n):
    arr = np.concatenate((source_rgb, target_rgb))
    eps1 = 1e-6
    # discretization as histogram
    min_v = arr.min() - eps
    max_v = arr.max() + eps
    xs = np.array([min_v + (max_v - min_v) * i / n for i in range(n + 1)])
    hist_in, _ = np.histogram(source_rgb, xs)
    hist_ref, _ = np.histogram(target_rgb, xs)
    xs = xs[:-1]
    # compute probability distribution
    cum_in = np.cumsum(hist_in)
    cum_ref = np.cumsum(hist_ref)
    d_in = cum_in / cum_in[-1]
    d_ref = cum_ref / cum_ref[-1]
    # tranfer
    t_d_in = np.interp(d_in, d_ref, xs)
    t_d_in[d_in <= d_ref[0]] = min_v
    t_d_in[d_in >= d_ref[-1]] = max_v
    arr_out = np.interp(source_rgb, xs, t_d_in)
    return arr_out

def one_dim_pdf_transfer_matlab(source_hist, target_hist):

    nbins = max(source_hist.shape)

    eps1 = 1e-6
    source_rgb_cumsum = np.cumsum(np.add(source_hist, eps1))
    source_rgb_cs = np.divide(source_rgb_cumsum, source_rgb_cumsum[-1])

    target_rgb_cumsum = np.cumsum(np.add(target_hist, eps1))
    target_rgb_cs = np.divide(target_rgb_cumsum, target_rgb_cumsum[-1])

    #  inversion
    xs = np.linspace(0, nbins-1, num=nbins)

    f = np.interp(source_rgb_cs, target_rgb_cs, xs)
    #f = interpolate.interp1d(interpolate.interp1d, target_rgb_cumsum, xs)
    f[source_rgb_cs <= target_rgb_cs[0]] = 0
    f[source_rgb_cs >= target_rgb_cs[-1]] = nbins - 1

    #if np.nansum(f) > 0:
    #    print('colour_transfer:pdf_transfer:NaN pdf_transfer has generated NaN values')

    return f

def Mean_CX(source_rgb, target_rgb, conversion):
    """
    compute using mean and standard deviation
    referencing from Color Transfer between Images 2001 by Erik Reinhard's paper
    http://erikreinhard.com/papers/colourtransfer.pdf
    :param source: source image in RGB color space (0-255) on numpy array
    :param target: target image in RGB color space (0-255) on numpy array
    :param conversion: two type color space conversions
                       'opencv' = opencv-python package
                       'matrix' = equation referencing from Color Transfer between Images by Erik Reinhard's paper
                             http://erikreinhard.com/papers/colourtransfer.pdf
    :return: output image in RGB color space (0-255) on numpy array
    """
    if conversion == 'opencv':
        return opencv_mean_cx(source_rgb, target_rgb)
    if conversion == 'matrix':
        return matrix_mean_cv(source_rgb, target_rgb)

def opencv_mean_cx(source_rgb, target_rgb):
    """
    Color transfer (CX) from target image's color characteristics into source image,
    using opencv-python package to convert between color space.
    :param source_rgb: source image in RGB color space (0-255) on numpy array
    :param target_rgb: target image in RGB color space (0-255) on numpy array
    :return: output_rgb: corrected image in RGB color space (0-255) on numpy array
    """
    # convert from RGB to LAB color space for source and target
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype("float32")
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype("float32")

    # statistics and color correction
    correction = color_correction(source_lab, target_lab, True)

    # convert from LAB to RGB color space
    output_rgb = cv2.cvtColor(correction.astype("uint8"), cv2.COLOR_LAB2RGB)

    return output_rgb

def matrix_mean_cv(source_rgb, target_rgb):
    """
    Color transfer (CX) from target image's color characteristics into source image,
    Referencing from Color Transfer between Images 2001 by Erik Reinhard's paper
    http://erikreinhard.com/papers/colourtransfer.pdf paper
    :param source_rgb: source image in RGB color space (0-255) on numpy array
    :param target_rgb: target image in RGB color space (0-255) on numpy array
    :return: output_rgb: corrected image in RGB color space (0-255) on numpy array
    """
    # convert from RGB to LAB color space for source and target
    source_lab = cx_rgb2lab(source_rgb, True)
    target_lab = cx_rgb2lab(target_rgb, True)

    # statistics and color correction
    correction = color_correction(source_lab, target_lab)

    # convert from LAB to RGB color space
    output_rgb = cx_lab2rgb(correction, True)

    return output_rgb

# Erik Reinhard 2001 Color Transfer between Images
def color_correction(source_lab, target_lab, clip='False'):
    """
    Color correction is to compute mean and standard deviation for each axis
    individually in the lab color space.
    Referencing from http://erikreinhard.com/papers/colourtransfer.pdf paper
    :param source_lab: source image in lab color space on numpy array
    :param target_lab: target image in lab color space on numpy array
    :param clip: limit the data points range (0-255) for opencv-python conversion lab to RGB
    :return: output_lab: corrected image in lab color space on numpy array
    """
    # split into individual channel space for statistics and color correction
    (source_l, source_a, source_b) = cv2.split(source_lab)
    (target_l, target_a, target_b) = cv2.split(target_lab)

    # equation 10 subtract source mean from the source data points
    l = source_l - source_l.mean()
    a = source_a - source_a.mean()
    b = source_b - source_b.mean()

    # equation 11 scale down by factoring the respective standard deviation
    l = (target_l.std() / source_l.std()) * l
    a = (target_a.std() / source_a.std()) * a
    b = (target_b.std() / source_b.std()) * b

    # adding the target mean back to the respective data points
    l = l + target_l.mean()
    a = a + target_a.mean()
    b = b + target_b.mean()

    # limit the data points range (0-255) for opencv-python conversion lab to RGB
    if clip:
        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)

    # merge individual channel back into lab color space
    output_lab = cv2.merge([l, a, b])

    return output_lab


