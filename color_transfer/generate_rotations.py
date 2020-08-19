# can remove if not in used
import numpy as np
import scipy.optimize
import cv2
from scipy.stats import special_ortho_group

eps = np.finfo(np.float).eps

def random_rotations(row=6, col=3):
    """
    generate orthogonal matrices for pdf transfer. Random rotation.
    :return:
    """
    assert row > 0
    rotation_matrices = [np.eye(col)]
    rotation_matrices.extend([np.matmul(rotation_matrices[0], special_ortho_group.rvs(dim=col)) for _ in range(row - 1)])
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

def pdf_transfer_test(source_rgb, target_rgb, optimal=True, n=300, step_size=1):
    """
    [Pitie05a] Pitie et al. N-Dimensional Probability Density Function Transfer and its Application to Colour Transfer. ICCV05.
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