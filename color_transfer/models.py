import cv2
import numpy as np
import scipy as sp

from color_transfer.utils import cx_rgb2lab, cx_lab2rgb, optimal_rotations, random_rotations
from color_transfer.generate_rotations import generate_rotations


eps = np.finfo(np.float32).eps

# The linear Monge-Kantorovitch linear colour mapping for
# example-based colour transfer. F. Pitié and A. Kokaram (2007) In 4th
# IEE European Conference on Visual Media Production (CVMP'07). London,
# November.

def Regrain(source_rgb, target_rgb, varargin):
    """
    https://github.com/pengbo-learn/python-color-transfer/blob/master/python_color_transfer/color_transfer.py
    :param source_rgb:
    :param target_rgb:
    :param varargin:
    :return:
    """
    numvarargs = len(varargin)
    if numvarargs > 1:
        print('regrain :TooManyInputs requires at most 1 optional input')
    optargs = np.ones(1)
    optargs[1: numvarargs] = varargin
    [smoothness] = optargs[:]

    source_rgb_regrain = source_rgb
    return regrain_rec(source_rgb_regrain, source_rgb, target_rgb, np.array([4, 16, 32, 64, 64, 64]), smoothness, 0)

def solve(source_rgb_regrain, source_rgb, target_rgb, nbits, smoothness, level):
    
    [k, vres, hres] = source_rgb.shape

    y0 = list(range(vres))
    y1 = list(range(vres))
    y2 = list(range(1, vres)) +  [vres - 1]
    y3 = list(range(vres))
    y4 = [0] + list(range(1, vres - 1))

    x0 = list(range(hres))
    x1 = list(range(1, hres)) + [vres - 1]
    x2 = list(range(hres))
    x3 = [0] + list(range(1, hres - 1))
    x4 = list(range(hres))

    source_rgb_g = source_rgb
    source_rgb_gx = (G0[:, [2:end end],:] - G0(:, [1 1: end - 1],:));
    source_rgb_gy = (G0([2:end end],:,:) - G0([1 1:end - 1],:,:));
    dI = sqrt(sum(G0x. ^ 2 + G0y. ^ 2, 3));


def regrain_rec()

def PDF_MKL_CX(source_rgb, target_rgb):
    """
    Probability Density Function (PDF) Monge-Kantorovitch linear (MKL) Color transfer (CX)
    :param source_rgb:
    :param target_rgb:
    :return:
    """
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

    return np.dot(source_rgb - mx0, t) + mx1

def PDF_MKL_CX_Matlab(source_rgb, target_rgb):
    if (source_rgb.ndim != 3):
        print('pictures must have 3 dimensions')
    # reshape images
    source_rgb_reshape = np.reshape(source_rgb, [], source_rgb.shape[2])
    target_rgb_reshape = np.reshape(target_rgb, [], target_rgb.shape[2])

    a = np.cov(source_rgb_reshape)
    b = np.cov(target_rgb_reshape)

    t = mkl(a, b)

    source_rgb_repmat = np.repmat(np.mean(source_rgb_reshape, axis=0), source_rgb_reshape.shape[0], 1)
    target_rgb_repmat = np.repmat(np.mean(target_rgb_reshape, axis=0), source_rgb_reshape.shape[0], 1)

    return np.reshape(np.add(np.dot(np.subtract(source_rgb_reshape, source_rgb_repmat), t), target_rgb_repmat), (source_rgb.shape[0], source_rgb.shape[1]))

def mkl(a, b):
    n = a.shape[0]
    da2, ua = np.linalg.eig(a)
    da = np.diag(np.sqrt(np.add(da2.clip(eps, None), eps)))
    c = np.dot(np.dot(np.dot(np.dot(da, ua.T), b), ua), da)
    uc, dc2 = np.linalg.eig(c)
    dc2 = np.dig(dc2)
    dc2.clip(eps, None)
    dc = np.diag(np.sqrt(np.add(dc2.clip(eps, None), eps)))
    da_inv = np.diag(1. / (np.diag(da)))
    t = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(ua, da_inv), uc), dc), uc.T), da_inv), ua.T)
    return t
# F. Pitie 2005 N-Dimensional PDF Transfer Iterative Distribution Transfer
def PDF_IDT_CX(source_rgb, target_rgb, bins=300, n_rot=10, relaxation=1):
    """
    Probability Density Function (PDF) Iterative Distribution Transfer (IDT) Color transfer (CX)
    :param source_rgb:
    :param target_rgb:
    :param bins:
    :param n_rot:
    :param relaxation:
    :return:
    """
    n_dims = source_rgb.shape[1]

    d0 = target_rgb.T
    d1 = target_rgb.T

    for i in range(n_rot):
        #  create a random orthonormal matrix
        r = sp.stats.special_ortho_group.rvs(n_dims).astype(np.float32)

        d0r = np.dot(r, d0)
        d1r = np.dot(r, d1)
        d_r = np.empty_like(d0)

        for j in range(n_dims):
            lo = min(d0r[j].min(), d1r[j].min())
            hi = max(d0r[j].max(), d1r[j].max())

            p0r, edges = np.histogram(d0r[j], bins=bins, range=[lo, hi])
            p1r, _ = np.histogram(d1r[j], bins=bins, range=[lo, hi])

            cp0r = p0r.cumsum().astype(np.float32)
            cp0r /= cp0r[-1]

            cp1r = p1r.cumsum().astype(np.float32)
            cp1r /= cp1r[-1]

            f = np.interp(cp0r, cp1r, edges[1:])

            d_r[j] = np.interp(d0r[j], edges[1:], f, left=0, right=bins)

        d0 = relaxation * np.linalg.solve(r, (d_r - d0r)) + d0

    return d0.T

def PDF_IDT_CX_Matlab(source_rgb, target_rgb, nb_iterations):
    if(source_rgb.ndim != 3):
        print('pictures must have 3 dimensions')
    nb_channels = source_rgb.shape[2]
    # reshape images as 3xN matrices
    source_rgb_reshape = []
    target_rgb_reshape = []
    for i in range(nb_channels):
        source_rgb_reshape[i, :] = np.reshape(source_rgb[:, :, i], (1, source_rgb.shape[0] * source_rgb.shape[1]))
        target_rgb_reshape[i, :] = np.reshape(target_rgb[:, :, i], (1, target_rgb.shape[0] * target_rgb.shape[1]))
    # building a sequence of (almost) random projections
    rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [2 / 3, 2 / 3, -1 / 3], [2 / 3, -1 / 3, 2 / 3], [-1 / 3, 2 / 3, 2 / 3]])
    rotation = np.zeros((nb_channels, rot.shape[0], rot.shape[1]))
    if(rotation == 'optimal'):
        rotation = optimal_rotations
    elif(rotation == 'random'):
        rotation[0, :] = rot
        for i in range(1, nb_iterations):
            rotation[i, :] = random_rotations()
    elif(rotation == 'generate'):
        rotation = generate_rotations(nb_channels, nb_iterations)
    else:
        rotation[0, :] = rot
        for i in range(1, nb_iterations):
            q, r = np.linalg.qr(np.random.random(size=(3, 3)))
            rotation[i, :] = np.matmul(rot, q)
    data_rotation = pdf_transfer(source_rgb_reshape, target_rgb_reshape, rotation, 1)
    result_rgb = target_rgb
    for i in range(nb_channels):
        result_rgb[:, :, i] = np.reshape(data_rotation[i, :], result_rgb.shape[0], result_rgb.shape[1])

    return result_rgb

def pdf_transfer(source_rgb, target_rgb, optimal=True, n=300, step_size=1):
    """
    :param source_rgb:
    :param target_rgb:
    :param optimal: True = Optimal rotation matrices from paper, False = Random rotation matrices.
    :param n:
    :param step_size:
    :return:
    """
    # reshape (h, w, c) to (c, h*w)
    [h, w, c] = source_rgb.shape
    reshape_arr_in = source_rgb.reshape(-1, c).transpose() / 255.
    reshape_arr_ref = target_rgb.reshape(-1, c).transpose() / 255.
    # pdf transfer
    # n times of 1d-pdf-transfer
    arr_out = np.array(reshape_arr_in)
    rotation_matrices = random_rotations
    if(optimal):
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
    img_arr_out = reshape_arr_out.transpose().reshape(h, w, c)
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
    for it in nb_iterations:
        print('IDT iteration ', it, '/', nb_iterations)
        r = rotations[it]
        nb_projs = r.shape[0]

        # apply rotation
        source_rgb_rot = np.matmul(r, source_rgb)
        target_rgb_rot = np.matmul(r, target_rgb)
        source_rgb_rot_ = np.zeros_like(source_rgb)

        # get the marginals, match them, and apply transformation
        for i in nb_projs:
            # get the data range
            datamin = min(min(source_rgb_rot[i, :]), min(target_rgb_rot[i, :])) - eps
            datamax = max(max(source_rgb_rot[i, :]), max(target_rgb_rot[i, :])) + eps
            u = np.array([i * (datamax - datamin) + datamin for i in range(n)])
            # get the projections
            source_rgb_projs = np.histogram(source_rgb_rot, u)
            target_rgb_projs = np.histogram(target_rgb_rot, u)
            # get the transport map
            f = one_dim_pdf_transfer(source_rgb_projs, target_rgb_projs, n)
            # apply the mapping
            source_rgb_rot_[i, :] = (np.interp(u, f.T, source_rgb_rot[i, :]) - 1) / (300-1) * (datamax-datamin) + datamin

        source_rgb = np.matmul(relaxation, (np.divide(r, np.subtract(source_rgb_rot_, source_rgb_rot)))) + source_rgb

    return source_rgb

# F. Pitie 2005 N-Dimensional PDF Transfer
def one_dim_pdf_transfer(source_rgb, target_rgb, n):
    arr = np.concatenate((source_rgb, target_rgb))
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

def one_dim_pdf_transfer_matlab(source_rgb, target_rgb):

    nbins = max(source_rgb.shape)

    source_rgb_cumsum = np.cumsum(source_rgb + eps)
    source_rgb_cumsum = np.divide(source_rgb_cumsum, source_rgb_cumsum[-1])

    target_rgb_cumsum = np.cumsum(target_rgb + eps)
    target_rgb_cumsum = np.divide(target_rgb_cumsum, target_rgb_cumsum[-1])

    #  inversion
    xs = np.linspace(0, nbins - 1, num=nbins)
    f = np.interp(xs, source_rgb_cumsum, target_rgb_cumsum);
    f[source_rgb_cumsum <= target_rgb_cumsum[0]] = 0
    f[source_rgb_cumsum >= target_rgb_cumsum[-1]] = nbins - 1;
    if np.nansum(f) > 0:
        print('colour_transfer:pdf_transfer:NaN pdf_transfer has generated NaN values');

    return f

def Mean_CX(source, target, conversion):
    """
    compute using mean and standard deviation
    referencing from Color Transfer between Images 2001 by Erik Reinhard's paper
    http://erikreinhard.com/papers/colourtransfer.pdf
    :param source: path and name of source image
    :param target: path and name of target image
    :param conversion: two type color space conversions
                       'opencv' = opencv-python package
                       'matrix' = equation referencing from Color Transfer between Images by Erik Reinhard's paper
                             http://erikreinhard.com/papers/colourtransfer.pdf
    :return: output image in RGB color space
    """
    source_bgr = cv2.imread(source, cv2.IMREAD_COLOR)
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_RGB2BGR)
    target_bgr = cv2.imread(target, cv2.IMREAD_COLOR)
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_RGB2BGR)
    if conversion == 'opencv':
        return opencv_cx(source_rgb, target_rgb)
    if conversion == 'matrix':
        return matrix_cv(source_rgb, target_rgb)

def opencv_cx(source_bgr, target_bgr):
    """
    Color transfer (CX) from target image's color characteristics into source image,
    using opencv-python package to convert between color space.
    :param source_bgr: source image in BGR color space on numpy array
    :param target_bgr: target image in BGR color space on numpy array
    :return: output_bgr: corrected image in BGR color space on numpy array
    """
    # convert from BGR to LAB color space for source and target
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype("float32")

    # statistics and color correction
    correction = color_correction(source_lab, target_lab)

    # convert from LAB to BGR color space
    output_bgr = cv2.cvtColor(correction.astype("uint8"), cv2.COLOR_LAB2BGR)

    return output_bgr

def matrix_cv(source_rgb, target_rgb):
    """
    Color transfer (CX) from target image's color characteristics into source image,
    Referencing from Color Transfer between Images 2001 by Erik Reinhard's paper
    http://erikreinhard.com/papers/colourtransfer.pdf paper
    :param source_rgb: source image in RGB color space on numpy array
    :param target_rgb: target image in RGB color space on numpy array
    :return: output_bgr: corrected image in BGR color space on numpy array
    """
    # convert from RGB to LAB color space for source and target
    source_lab = cx_rgb2lab(source_rgb, True)
    target_lab = cx_rgb2lab(target_rgb, True)

    # statistics and color correction
    correction = color_correction(source_lab, target_lab, True)

    # convert from LAB to RGB color space
    output_rgb = cx_lab2rgb(correction, True)
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

    return output_bgr

# Erik Reinhard 2001 Color Transfer between Images
def color_correction(source_lab, target_lab, clip='False'):
    """
    Color correction is to compute mean and standard deviation for each axis
    individually in the lab color space.
    Referencing from http://erikreinhard.com/papers/colourtransfer.pdf paper
    :param source_lab: source image in lab color space on numpy array
    :param target_lab: target image in lab color space on numpy array
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

    # limit the data points range for opencv-python conversion lab to BGR
    if(clip):
        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)

    # merge individual channel back into lab color space
    output_lab = cv2.merge([l, a, b])

    return output_lab


