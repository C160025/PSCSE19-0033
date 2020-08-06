import cv2
import numpy as np
from scipy.stats import special_ortho_group
from color_transfer.utils import cx_rgb2lab, cx_lab2rgb

eps = np.finfo(np.float32).eps

# The linear Monge-Kantorovitch linear colour mapping for
# example-based colour transfer. F. Pitié and A. Kokaram (2007) In 4th
# IEE European Conference on Visual Media Production (CVMP'07). London,
# November.

def MKL_CX(source_rgb, target_rgb):
    """
    [Pitie07b] Pitie et al. The linear Monge-Kantorovitch linear colour mapping for example-based colour transfer. CVMP07.
    Probability Density Function (PDF) Monge-Kantorovitch linear (MKL) Colour transfer (CX)
    :param source_rgb: source image in RGB colour space (0-255) on numpy array
    :param target_rgb: target image in RGB colour space (0-255) on numpy array
    :return: output_rgb: corrected image in RGB colour space (0-255) on numpy array
    """
    if source_rgb.ndim != 3 and target_rgb.ndim != 3:
        print('pictures must have 3 dimensions')
    vres, hres, dim = source_rgb.shape
    source = (source_rgb / 255.).reshape(-1, dim)
    target = (target_rgb / 255.).reshape(-1, dim)

    source_cov = np.cov(source.T)
    target_cov = np.cov(target.T)

    t = mkl(source_cov, target_cov)

    mx0 = np.mean(source, axis=0)
    mx1 = np.mean(target, axis=0)

    xr = (source - mx0) @ t + mx1

    return np.multiply(xr.reshape(source_rgb.shape), 255).astype(np.uint8)

def mkl(source_cov, target_cov):

    da2, ua = np.linalg.eig(source_cov)
    da = np.diag(np.sqrt(da2.clip(eps, None)))

    c = da @ ua.T @ target_cov @ ua @ da

    dc2, uc = np.linalg.eig(c)
    dc = np.diag(np.sqrt(dc2.clip(eps, None)))

    da_inv = np.diag(1. / (np.diag(da)))

    return ua @ da_inv @ uc @ dc @ uc.T @ da_inv @ ua.T

# F. Pitie 2007 Automated Colour Grading using Colour Distribution Transfer
def REGRAIN_CX(source_rgb, target_rgb):
    """
    referencing from the following papers :
    [Pitie05b] Pitie et al. Towards Automated Colour Grading. CVMP05.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
    [Pitie07a] Pitie et al. Automated colour grading using colour distribution transfer. CVIU. 2007.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
    :param source_rgb:
    :param target_rgb:
    :return:
    """
    if source_rgb.ndim != 3 and target_rgb.ndim != 3:
        print('pictures must have 3 dimensions')
    idt_rgb = IDT_CX(source_rgb, target_rgb, bins=300, nb_iterations=30, relaxation=1)
    return regrain_cx(source_rgb, idt_rgb)

def regrain_cx(source_rgb, idt_rgb):
    """
    referencing from the following papers :
    [Pitie05b] Pitie et al. Towards Automated Colour Grading. CVMP05.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
    [Pitie07a] Pitie et al. Automated colour grading using colour distribution transfer. CVIU. 2007.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
    :param source_rgb:
    :param idt_rgb:
    :return:
    """
    source = source_rgb / 255.
    idt = idt_rgb / 255.

    source_zero = np.zeros_like(source_rgb)
    output = regrain_rec(source_zero, source, idt, nb_bits=np.array([4, 16, 32, 64, 64, 64]), smoothness=1, level=0)
    return (output * 255.).astype(np.uint8)

def solve(result, source, idt, n_bits, smoothness, level):
    """
    referencing from the following papers :
    [Pitie05b] Pitie et al. Towards Automated Colour Grading. CVMP05.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
    [Pitie07a] Pitie et al. Automated colour grading using colour distribution transfer. CVIU. 2007.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
    :param result:
    :param source:
    :param idt:
    :param n_bits:
    :param smoothness:
    :param level:
    :return:
    """
    [_, _, k] = source.shape

    g = source

    gx1 = np.concatenate((g[:, 1:, :], g[:, [-1], :]), axis=1)
    gx2 = np.concatenate((g[:, [0], :], g[:, 0:-1, :]), axis=1)
    gy1 = np.concatenate((g[1:, :, :], g[[-1], :, :]), axis=0)
    gy2 = np.concatenate((g[[0], :, :], g[0:-1, :, :]), axis=0)
    gx = gx1 - gx2
    gy = gy1 - gy2
    dI = np.sqrt(np.sum((np.add(gx**2, gy**2)), axis=2))

    h = 2 ** (-level)
    # equation 14 limit the stretching during the transformation
    psi = (256. * dI / 5.).clip(None, 1)
    # equation 12 limit the flat areas remain flats
    phi = 30. / (1 + 10 * dI / max(smoothness, eps)) * h

    # construct lambda calculus function to approximate the four neighbour pixels
    p1 = lambda arr: np.concatenate((arr[:, 1:], arr[:, -1:]), axis=1)
    p2 = lambda arr: np.concatenate((arr[1:, :], arr[-1:, :]), axis=0)
    p3 = lambda arr: np.concatenate((arr[:, :1], arr[:, :-1]), axis=1)
    p4 = lambda arr: np.concatenate((arr[:1, :], arr[:-1, :]), axis=0)

    # equation 19
    phi1 = (p1(phi) + phi) / 2
    phi2 = (p2(phi) + phi) / 2
    phi3 = (p3(phi) + phi) / 2
    phi4 = (p4(phi) + phi) / 2

    rho = 1/5
    # equation 18
    for i in range(n_bits):
        den = psi + phi1 + phi2 + phi3 + phi4
        num = (np.repeat(psi[:, :, np.newaxis], k, axis=2) * idt
               + np.repeat(phi1[:, :, np.newaxis], k, axis=2) * (p1(result) - p1(source) + source)
               + np.repeat(phi2[:, :, np.newaxis], k, axis=2) * (p2(result) - p2(source) + source)
               + np.repeat(phi3[:, :, np.newaxis], k, axis=2) * (p3(result) - p3(source) + source)
               + np.repeat(phi4[:, :, np.newaxis], k, axis=2) * (p4(result) - p4(source) + source))
        result = num / np.repeat(den[:, :, np.newaxis], k, axis=2) * (1 - rho) + rho * result

    return result

def regrain_rec(source_zero, source, idt, nb_bits, smoothness, level):
    """
    referencing from the following papers :
    [Pitie05b] Pitie et al. Towards Automated Colour Grading. CVMP05.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
    [Pitie07a] Pitie et al. Automated colour grading using colour distribution transfer. CVIU. 2007.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
    :param source_zero:
    :param source:
    :param idt:
    :param n_bits:
    :param smoothness:
    :param level:
    :return:
    """
    vres, hres, _ = source.shape

    vres2 = int(np.ceil(vres / 2))
    hres2 = int(np.ceil(hres / 2))

    if len(n_bits) > 1 and vres2 > 20 and hres2 > 20:
        source_resize = cv2.resize(source, (hres2, vres2), interpolation=cv2.INTER_LINEAR)
        idt_resize = cv2.resize(idt, (hres2, vres2), interpolation=cv2.INTER_LINEAR)
        source_zero_resize = cv2.resize(source_zero, (hres2, vres2), interpolation=cv2.INTER_LINEAR)
        source_zero_resize = regrain_rec(source_zero_resize, source_resize, idt_resize, n_bits[1:], smoothness, level=level+1)
        result = cv2.resize(source_zero_resize, (hres, vres), interpolation=cv2.INTER_LINEAR)

    return solve(result, source, idt, nb_bits[0], smoothness, level)

# F. Pitie 2005 N-Dimensional PDF Transfer Iterative Distribution Transfer
def IDT_CX(source_rgb, target_rgb, bins=300, nb_iterations=30, relaxation=1):
    """
    Probability Density Function (PDF) Iterative Distribution Transfer (IDT) Colour transfer (CX)
    referencing from the following papers :
    [Pitie05a] Pitie et al. N-Dimensional Probability Density Function Transfer and its Application to Colour Transfer.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05iccv.pdf
    [Pitie05b] Pitie et al. Towards Automated Colour Grading. CVMP05.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
    [Pitie07a] Pitie et al. Automated colour grading using colour distribution transfer. CVIU. 2007.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
    :param source_rgb: source in RGB on numpy array
    :param target_rgb: target in RGB on numpy array
    :param bins: bandwidth size for the distribution
    :param nb_iterations: number of repetition to compute the transfer mapping
    :param relaxation: integrality gap for approximation
    :return: transferred result in RGB on numpy array
    """
    if source_rgb.ndim != 3 and target_rgb.ndim != 3:
        print('pictures must have 3 dimensions')

    vres, hres, dim = source_rgb.shape

    # reshape images by flatten RGB array
    source_reshape = source_rgb.T.reshape(dim, -1) / 255.
    target_reshape = target_rgb.T.reshape(dim, -1) / 255.

    # implementation of N-Dimensional PDF Transfer
    source = pdf_transfer(source_reshape, target_reshape, bins, nb_iterations, relaxation)

    # reshape image back to normal RGB array
    output_r = source[0, :].reshape((hres, vres)).T * 255.
    output_g = source[1, :].reshape((hres, vres)).T * 255.
    output_b = source[2, :].reshape((hres, vres)).T * 255.

    return cv2.merge([output_r, output_g, output_b]).astype(np.uint8)

def pdf_transfer(source_reshape, target_reshape, bins, nb_iterations, relaxation):
    """
    referencing from the following papers :
    [Pitie05a] Pitie et al. N-Dimensional Probability Density Function Transfer and its Application to Colour Transfer.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05iccv.pdf
    [Pitie05b] Pitie et al. Towards Automated Colour Grading. CVMP05.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
    [Pitie07a] Pitie et al. Automated colour grading using colour distribution transfer. CVIU. 2007.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
    :param source_reshape: source in flatten RGB on numpy array
    :param target_reshape: target in flatten RGB on numpy array
    :param bins: bandwidth size for the distribution
    :param nb_iterations: number of repetition to compute the transfer mapping
    :param relaxation: integrality gap for approximation
    :return: remapped result flatten RGB on numpy array
    """
    source = source_reshape
    n_dim = source_reshape.shape[0]

    # simple implementation of N-Dimensional PDF Transfer
    first_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                               [2/3, 2/3, -1/3], [2/3, -1/3, 2/3], [-1/3, 2/3, 2/3]])
    for i in range(nb_iterations):
        #  generate random orthogonal rotation matrix
        orthogonal_matrix = special_ortho_group.rvs(n_dim).astype(np.float32)
        rotation = first_rotation @ orthogonal_matrix if i > 0 else first_rotation

        # apply rotation to change the coordinate
        source_rotation = rotation @ source
        target_rotation = rotation @ target_reshape
        source_rotation_ = np.empty_like(source_rotation)
        nb_projections = rotation.shape[0]

        # get the marginals, match them, and apply transformation
        for j in range(nb_projections):
            # get the data plotting range
            data_min = min(source_rotation[j].min(), target_rotation[j].min()) - eps
            data_max = max(source_rotation[j].max(), target_rotation[j].max()) + eps

            # projection the source and target along the axis
            source_projections, edges = np.histogram(source_rotation[j], bins=bins - 1,
                                                     range=[data_min, data_max])
            target_projections, _ = np.histogram(target_rotation[j], bins=bins - 1,
                                                 range=[data_min, data_max])

            # transport map on 1-Dimensional PDF Transfer
            discrete_var = one_d_pdf_transfer(source_projections, target_projections, edges)

            # apply the mapping
            source_rotation_[j] = np.interp(source_rotation[j], edges[1:], discrete_var, left=0,
                                            right=bins - 1)

        # apply this iteration transformation result
        source = relaxation * np.linalg.pinv(rotation) @ (source_rotation_ - source_rotation) + source
    return source

def one_d_pdf_transfer(source_projections, target_projections, edges):
    """
    transport map on 1-Dimensional PDF Transfer small damping term that facilitate the inversion
    referencing from the following papers :
    [Pitie05a] Pitie et al. N-Dimensional Probability Density Function Transfer and its Application to Colour Transfer.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05iccv.pdf
    [Pitie05b] Pitie et al. Towards Automated Colour Grading. CVMP05.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
    [Pitie07a] Pitie et al. Automated colour grading using colour distribution transfer. CVIU. 2007.
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
    :param source_projections: source histogram data
    :param target_projections: target histogram data
    :param edges: bin edges to pass into the 1-D linear interpolation
    :return: discrete lookup table result
    """
    # small damping to approximate the cumulative pdf
    eps_6 = 1e-6

    # cumulative pdf of source
    source_cum_projections = (source_projections + eps_6).cumsum()
    source_damp = source_cum_projections / source_cum_projections[-1]

    # cumulative pdf of target
    target_cum_projections = (target_projections + eps_6).cumsum()
    target_damp = target_cum_projections / target_cum_projections[-1]

    # 1-D linear interpolation for discrete data points
    return np.interp(source_damp, target_damp, edges[1:])

def Mean_CX(source_rgb, target_rgb, conversion):
    """
    compute using mean and standard deviation
    referencing from Color Transfer between Images 2001 by Erik Reinhard's paper
    http://erikreinhard.com/papers/colourtransfer.pdf
    :param source: source in RGB colour space (0-255) on numpy array
    :param target: target in RGB colour space (0-255) on numpy array
    :param conversion: two type colour space conversions
                       'opencv' = opencv-python package
                       'matrix' = equation referencing from Color Transfer between Images by Erik Reinhard's paper
                             http://erikreinhard.com/papers/colourtransfer.pdf
    :return: output image in RGB colour space (0-255) on numpy array
    """
    if source_rgb.ndim != 3 and target_rgb.ndim != 3:
        print('pictures must have 3 dimensions')
    if conversion == 'opencv':
        return opencv_mean_cx(source_rgb, target_rgb)
    if conversion == 'matrix':
        return matrix_mean_cx(source_rgb, target_rgb)

def opencv_mean_cx(source_rgb, target_rgb):
    """
    Colour transfer (CX) from target image's colour characteristics into source image,
    using opencv-python package to convert between colour space.
    :param source_rgb: source in RGB colour space (0-255) on numpy array
    :param target_rgb: target in RGB colour space (0-255) on numpy array
    :return: corrected image in RGB colour space (0-255) on numpy array
    """
    # convert from RGB to LAB colour space for source and target
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float)

    # statistics and u correction
    correction = colour_correction(source_lab, target_lab, True)
    #correction = colour_correction(source_rgb, target_rgb, True)

    # convert from LAB to RGB colour space
    return cv2.cvtColor(correction.astype(np.uint8), cv2.COLOR_LAB2RGB)
    # return correction.astype(np.uint8)

def matrix_mean_cx(source_rgb, target_rgb):
    """
    Colour transfer (CX) from target image's colour characteristics into source image,
    Referencing from Color Transfer between Images 2001 by Erik Reinhard's paper
    http://erikreinhard.com/papers/colourtransfer.pdf
    :param source_rgb: source in RGB colour space (0-255) on numpy array
    :param target_rgb: target in RGB colour space (0-255) on numpy array
    :return: corrected image in RGB colour space (0-255) on numpy array
    """
    # convert from RGB to LAB colour space for source and target
    source_lab = cx_rgb2lab(source_rgb, True)
    target_lab = cx_rgb2lab(target_rgb, True)

    # statistics and colour correction
    correction = colour_correction(source_lab, target_lab)
    # correction = colour_correction(source_rgb, target_rgb)

    # convert from LAB to RGB colour space
    return cx_lab2rgb(correction, True)
    # return correction.astype(np.uint8)

# Erik Reinhard 2001 Color Transfer between Images
def colour_correction(source_lab, target_lab, clip='False'):
    """
    Colour correction is to compute mean and standard deviation for each axis
    individually in the lab colour space.
    Referencing from Color Transfer between Images 2001 by Erik Reinhard's paper
    http://erikreinhard.com/papers/colourtransfer.pdf
    :param source_lab: source in lab colour space on numpy array
    :param target_lab: target in lab colour space on numpy array
    :param clip: limit the data points range (0-255) for opencv-python conversion lab to RGB
    :return: corrected image in lab colour space on numpy array
    """
    # split into individual channel space for statistics and colour correction
    (source_l, source_a, source_b) = cv2.split(source_lab)
    (target_l, target_a, target_b) = cv2.split(target_lab)

    # equation 10/2.4 subtract source mean from the source data points
    l = source_l - source_l.mean()
    a = source_a - source_a.mean()
    b = source_b - source_b.mean()

    # equation 11/2.5 scale down by factoring the respective standard deviation
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

    # merge individual channel back into lab colour space
    return cv2.merge([l, a, b])
