"""
Analysis of several calibration runs agains all available 
chessboard images.
"""

import numpy as np
from .planefit import fit_plane


def apply_metric_to_all_point_clouds(triang, metric_func):

    n_calibs, n_images, _, _ = triang.shape

    res = np.zeros((n_calibs, n_images))

    for i in range(n_calibs):
        for j in range(n_images):

            pcloud = triang[i, j]
            plane_coefs = fit_plane(pcloud)

            res[i, j] = metric_func(plane_coefs, pcloud)

    return res


def get_list_of_2d_index_pairs(n_rows, n_cols):
    return [(i, j) for i in range(n_rows) for j in range(n_cols)]


def flatten_and_sort_metric_matrix(mat):

    n_calibs, n_images = mat.shape

    mat_values = mat.ravel()
    index_pairs = get_list_of_2d_index_pairs(n_calibs, n_images)

    data = zip(mat_values, index_pairs)

    data_s = sorted(data, key=lambda entry: entry[0])

    return data_s
