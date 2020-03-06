import numpy as np


def fit_plane(pcloud):

    ones = np.ones((pcloud.shape[0], 1))
    A = np.hstack((pcloud, ones))

    U, s, V = np.linalg.svd(A)
    x = V[-1, :]

    return x


def plane_z(plane_coefs, x, y):

    a, b, c, d = plane_coefs

    z = (-a * x - b * y - d) / c

    return z


def rms_diff(x1, x2):

    sq_diffs = (x1 - x2)**2
    mse = np.mean(sq_diffs)
    return np.sqrt(mse)


def plane_diff_rms(plane_coefs, pcloud):

    x = pcloud[:, 0]
    y = pcloud[:, 1]
    z = pcloud[:, 2]

    z_hat = plane_z(plane_coefs, x, y)

    return rms_diff(z, z_hat)
