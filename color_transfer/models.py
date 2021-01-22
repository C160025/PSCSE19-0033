import cv2
import numpy as np
from scipy.stats import special_ortho_group
from color_transfer.utils import cx_rgb2lab, cx_lab2rgb

# machine epsilon for floating point arithmetic
eps = np.finfo(np.float32).eps

# F. Pitie 2007 Monge-Kantorovitch linear Colour transfer
def MKL_CX(source_rgb, target_rgb):
    """
    Monge-Kantorovitch linear (MKL) Colour transfer (CX)
    [Pitie07b] Pitie et al. 2007 The linear Monge-Kantorovitch linear colour mapping for example-based colour transfer
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cvmp.pdf
    :param source_rgb: source in RGB on numpy array
    :param target_rgb: target in RGB on numpy array
    :return: output in RGB colour space on numpy array
    """
    if source_rgb.ndim != 3 and target_rgb.ndim != 3:
        print('pictures must have 3 dimensions')
    _, _, ch = source_rgb.shape

    # flatten and normalized RGB array
    source = (source_rgb / 255.).reshape(-1, ch)
    target = (target_rgb / 255.).reshape(-1, ch)

    # estimate the covariance matrix
    source_cov = np.cov(source.T)
    target_cov = np.cov(target.T)

    # transfer the colour mapping with Monge-Kantorovitch linear
    transfer = mkl(source_cov, target_cov)

    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)

    result = (source - source_mean) @ transfer + target_mean

    # unflatten and denormalized RGB array
    return (result.reshape(source_rgb.shape) * 255).astype(np.uint8)

def mkl(source_cov, target_cov):
    """
    Monge-Kantorovitch linear transfer the source to colour map target
    :param source_cov: flatten source covariance in RGB on numpy array
    :param target_cov: flatten target covariance in RGB on numpy array
    :return: flatten result in RGB on numpy array
    """

    # compute eigenvalues and eigenvectors of source covariance
    source_eigval, source_eigvec = np.linalg.eig(source_cov)
    # extract diagonal matrix from source eigenvalues
    source_diag = np.diag(np.sqrt(source_eigval.clip(eps, None)))

    # displacement mapping cost between the source and target
    cost = source_diag @ source_eigvec.T @ target_cov @ source_eigvec @ source_diag

    # compute eigenvalues and eigenvectors of the displacement cost
    cost_eigval, cost_eigvec = np.linalg.eig(cost)
    # extract diagonal matrix from cost eigenvalues
    cost_diag = np.diag(np.sqrt(cost_eigval.clip(eps, None)))
    # inverse the source diagonal matrix
    source_diag_inv = np.diag(1. / (np.diag(source_diag)))
    # transformation solution
    return source_eigvec @ source_diag_inv @ cost_eigvec @ cost_diag \
           @ cost_eigvec.T @ source_diag_inv @ source_eigvec.T

# F. Pitie 2007 Regain Colour Transfer to Reduce Gain Noise Artefacts
def REGRAIN_CX(source_rgb, target_rgb):
    """
    Process Iterative Distribution Transfer (IDT) Colour transfer (CX) and follow by Regain Colour transfer (CX)
    referencing from the following papers :
    [Pitie05b] Pitié et al. 2005 Towards Automated Colour Grading
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
    [Pitie07a] Pitié et al. 2007 Automated colour grading using colour distribution transfer
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
    :param source_rgb: source in RGB on numpy array
    :param target_rgb: target in RGB on numpy array
    :return: transferred result in RGB on numpy array
    """
    if source_rgb.ndim != 3 and target_rgb.ndim != 3:
        print('pictures must have 3 dimensions')
    idt_rgb = IDT_CX(source_rgb, target_rgb, bins=300, nb_iterations=30, relaxation=1)
    return regrain_cx(source_rgb, idt_rgb)

def regrain_cx(source_rgb, idt_rgb):
    """
    Regain Colour transfer (CX) on IDT result to match the gradient level to the original source image
    referencing from the following papers :
    [Pitie05b] Pitié et al. 2005 Towards Automated Colour Grading
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
    [Pitie07a] Pitié et al. 2007 Automated colour grading using colour distribution transfer
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
    :param source_rgb: source in RGB on numpy array
    :param idt_rgb: IDT result in RGB on numpy array
    :return: transferred result in RGB on numpy array
    """
    # normalized RGB array
    source = source_rgb / 255.
    idt = idt_rgb / 255.
    source_zero = np.zeros_like(source_rgb)

    # process the regrain recursion
    output = regrain_recursion(source_zero, source, idt, n_bits=np.array([4, 16, 32, 64, 64, 64]), smoothness=1, level=0)

    # denormalized RGB array
    return (output * 255.).astype(np.uint8)

def regrain_recursion(source_zero, source, idt, n_bits, smoothness, level):
    """
    resize images size by half using bilinear interpolation to smooth out the edges or noises
    referencing from the following papers :
    [Pitie05b] Pitié et al. 2005 Towards Automated Colour Grading
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
    [Pitie07a] Pitié et al. 2007 Automated colour grading using colour distribution transfer
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
    :param source_zero: source with all zeros in RGB on numpy array
    :param source: source in RGB on numpy array
    :param idt: IDT result in RGB on numpy array
    :param n_bits: number of bits to control the resize
    :param smoothness: approximate forecast to reduce the noise in the data
    :param level: incremental recursion level
    :return: solved result in RGB on numpy array
    """
    hgt, wdth, _ = source.shape

    hgt2 = int(np.ceil(hgt / 2))
    wdth2 = int(np.ceil(wdth / 2))

    # recurring resize the image size by half
    if len(n_bits) > 1 and hgt2 > 20 and wdth2 > 20:
        source_resize = cv2.resize(source, (wdth2, hgt2), interpolation=cv2.INTER_LINEAR)
        idt_resize = cv2.resize(idt, (wdth2, hgt2), interpolation=cv2.INTER_LINEAR)
        source_zero_resize = cv2.resize(source_zero, (wdth2, hgt2), interpolation=cv2.INTER_LINEAR)
        source_zero_resize = regrain_recursion(source_zero_resize, source_resize, idt_resize, n_bits[1:], smoothness, level=level+1)
        source_zero = cv2.resize(source_zero_resize, (wdth, hgt), interpolation=cv2.INTER_LINEAR)

    # solve with resize image
    return solve(source_zero, source, idt, n_bits[0], smoothness, level)

def solve(result, source, idt, n_bits, smoothness, level):
    """
    referencing from the following papers :
    [Pitie05b] Pitié et al. 2005 Towards Automated Colour Grading
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
    [Pitie07a] Pitié et al. 2007 Automated colour grading using colour distribution transfer
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
    :param result: recursion result in RGB on numpy array
    :param source: source in RGB on numpy array
    :param idt: IDT result in RGB on numpy array
    :param n_bits: number of bits to control the loop
    :param smoothness: approximate forecast to reduce the noise in the data
    :param level: incremental recursion level
    :return: result in RGB on numpy array
    """
    _, _, ch = source.shape

    # construct lambda calculus function to approximate the four neighbour pixels
    pixel_1 = lambda arr: np.concatenate((arr[:, 1:], arr[:, -1:]), axis=1)
    pixel_2 = lambda arr: np.concatenate((arr[1:, :], arr[-1:, :]), axis=0)
    pixel_3 = lambda arr: np.concatenate((arr[:, :1], arr[:, :-1]), axis=1)
    pixel_4 = lambda arr: np.concatenate((arr[:1, :], arr[:-1, :]), axis=0)

    delta_x = pixel_3(idt) - pixel_1(idt)
    delta_y = pixel_4(idt) - pixel_2(idt)
    delta = np.sqrt(np.sum((np.add(delta_x**2, delta_y**2)), axis=2))

    h = 2 ** (-level)
    # equation 2.13 weight field that limit the stretching during the transformation
    psi = (256. * delta / 5.).clip(None, 1)
    # equation 2.12 weight field that limit the flat areas remain flats
    phi = (30. * h) / (1 + 10 * delta / max(smoothness, eps))
    # partial of equation 2.20
    phi_1 = (pixel_1(phi) + phi) / 2
    phi_2 = (pixel_2(phi) + phi) / 2
    phi_3 = (pixel_3(phi) + phi) / 2
    phi_4 = (pixel_4(phi) + phi) / 2

    rho = 1/5
    # equation 2.18 Linear Elliptic Partial Differential
    for i in range(n_bits):
        den = psi + phi_1 + phi_2 + phi_3 + phi_4
        # partial of equation 2.20
        num = (np.repeat(psi[:, :, np.newaxis], ch, axis=2) * idt
               + np.repeat(phi_1[:, :, np.newaxis], ch, axis=2) * (pixel_1(result) - pixel_1(source) + source)
               + np.repeat(phi_2[:, :, np.newaxis], ch, axis=2) * (pixel_2(result) - pixel_2(source) + source)
               + np.repeat(phi_3[:, :, np.newaxis], ch, axis=2) * (pixel_3(result) - pixel_3(source) + source)
               + np.repeat(phi_4[:, :, np.newaxis], ch, axis=2) * (pixel_4(result) - pixel_4(source) + source))
        result = num / np.repeat(den[:, :, np.newaxis], ch, axis=2) * (1 - rho) + rho * result

    return result


# François Pitié 2005 N-Dimensional PDF Transfer / Iterative Distribution Transfer
def IDT_CX(source_rgb, target_rgb, bins=300, nb_iterations=30, relaxation=1):
    """
    Probability Density Function (PDF) or Iterative Distribution Transfer (IDT) Colour transfer (CX)
    referencing from the following papers :
    [Pitie05a] Pitié et al. 2005 N-Dimensional Probability Density Function Transfer and its Application to Colour Transfer
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05iccv.pdf
    [Pitie05b] Pitié et al. 2005 Towards Automated Colour Grading
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
    [Pitie07a] Pitié et al. 2007 Automated colour grading using colour distribution transfer
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

    hgt, wdth, ch = source_rgb.shape

    # reshape images by flatten and normalized RGB array
    source_reshape = source_rgb.T.reshape(ch, -1) / 255.
    target_reshape = target_rgb.T.reshape(ch, -1) / 255.

    # facilitate the PDF/IDT Transfer
    source = pdf_transfer(source_reshape, target_reshape, bins, nb_iterations, relaxation)

    # reshape image back by unflatten and denormalized RGB array
    output_r = source[0, :].reshape((wdth, hgt)).T * 255.
    output_g = source[1, :].reshape((wdth, hgt)).T * 255.
    output_b = source[2, :].reshape((wdth, hgt)).T * 255.

    return cv2.merge([output_r, output_g, output_b]).astype(np.uint8)

def pdf_transfer(source_reshape, target_reshape, bins, nb_iterations, relaxation):
    """
    decreases the Kullback-Leibler (KL) divergence measurement till the iterations end
    referencing from the following papers :
    [Pitie05a] Pitié et al. 2005 N-Dimensional Probability Density Function Transfer and its Application to Colour Transfer
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05iccv.pdf
    [Pitie05b] Pitié et al. 2005 Towards Automated Colour Grading
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
    [Pitie07a] Pitié et al. 2007 Automated colour grading using colour distribution transfer
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie07cviu.pdf
    :param source_reshape: source in flatten RGB on numpy array
    :param target_reshape: target in flatten RGB on numpy array
    :param bins: bandwidth size for the distribution
    :param nb_iterations: number of repetition to compute the transfer mapping
    :param relaxation: integrality gap for approximation
    :return: remapped result flatten RGB on numpy array
    """
    source = source_reshape
    hgt, _ = source_reshape.shape

    # initialize the fist rotation matrix
    first_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                               [2/3, 2/3, -1/3], [2/3, -1/3, 2/3], [-1/3, 2/3, 2/3]])
    for i in range(nb_iterations):
        #  generate random orthogonal rotation matrix
        orthogonal_matrix = special_ortho_group.rvs(hgt).astype(np.float32)
        rotation = first_rotation @ orthogonal_matrix if i > 0 else first_rotation

        # apply rotation to change the coordinate
        source_rotation = rotation @ source
        target_rotation = rotation @ target_reshape
        source_rotation_ = np.empty_like(source_rotation)
        nb_projections, _ = rotation.shape

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

        # apply transformation result
        source = relaxation * np.linalg.pinv(rotation) @ (source_rotation_ - source_rotation) + source
    return source

def one_d_pdf_transfer(source_projections, target_projections, edges):
    """
    transport map on 1-Dimensional PDF Transfer small damping term that facilitate the inversion
    referencing from the following papers :
    [Pitie05a] Pitié et al. 2005 N-Dimensional Probability Density Function Transfer and its Application to Colour Transfer
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05iccv.pdf
    [Pitie05b] Pitié et al. 2005 Towards Automated Colour Grading
    https://github.com/frcs/colour-transfer/blob/master/publications/pitie05cvmp.pdf
    [Pitie07a] Pitié et al. 2007 Automated colour grading using colour distribution transfer
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

# Erik Reinhard 2001 Color Transfer between Images
def Mean_CX(source_rgb, target_rgb, conversion):
    """
    compute using mean and standard deviation
    referencing from Reinhard et al. 2001 Color Transfer between Images
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
    if conversion == 'noconv':
        return noconv_mean_cx(source_rgb, target_rgb)

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

    # statistics and colour correction
    correction = colour_correction(source_lab, target_lab, True)

    # convert from LAB to RGB colour space
    return cv2.cvtColor(correction.astype(np.uint8), cv2.COLOR_LAB2RGB)

def matrix_mean_cx(source_rgb, target_rgb):
    """
    Colour transfer (CX) from target image's colour characteristics into source image,
    referencing from Reinhard et al. 2001 Color Transfer between Images
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

    # convert from LAB to RGB colour space
    return cx_lab2rgb(correction, True)

def noconv_mean_cx(source_rgb, target_rgb):
    """
    Colour transfer (CX) from target image's colour characteristics into source image,
    colour transfer directly on the RGB colour space.
    :param source_rgb: source in RGB colour space (0-255) on numpy array
    :param target_rgb: target in RGB colour space (0-255) on numpy array
    :return: corrected image in RGB colour space (0-255) on numpy array
    """

    # statistics and colour correction
    correction = colour_correction(source_rgb, target_rgb)

    # no colour space conversion
    return correction.astype(np.uint8)

def colour_correction(source_lab, target_lab, clip='False'):
    """
    Colour correction is to compute mean and standard deviation,
    individually in for each axis under lab colour space.
    referencing from Reinhard et al. 2001 Color Transfer between Images
    http://erikreinhard.com/papers/colourtransfer.pdf
    :param source_lab: source in lab colour space on numpy array
    :param target_lab: target in lab colour space on numpy array
    :param clip: limit the data points range (0-255) for opencv-python conversion lab to RGB
    :return: corrected image in lab colour space on numpy array
    """
    # split into individual channel space for statistics and colour correction to optimize the computation time
    (source_l, source_a, source_b) = cv2.split(source_lab)
    (target_l, target_a, target_b) = cv2.split(target_lab)

    # equation 2.4 subtract source mean from the source data points
    l = source_l - source_l.mean()
    a = source_a - source_a.mean()
    b = source_b - source_b.mean()

    # equation 2.5 scale down by factoring the respective standard deviation
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
