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


def detect_good_triangulations(res_np, target, tol):

    bottom = target - tol
    top = target + tol

    return (res_np > bottom) & (res_np < top)


def find_best_calib_based_on_triangulations(good):

    ngood_per_calib = np.sum(good, axis=1)
    best_calib_run_idx = ngood_per_calib.argmax()
    good_image_indices_in_best_calib = np.nonzero(good[best_calib_run_idx])[0]

    return ngood_per_calib, best_calib_run_idx, good_image_indices_in_best_calib


def analyze_good_tringulations(good):

    calib_runs_with_nonzero_good = dict()

    for calib_idx in range(len(good)):
        calib_metrics = good[calib_idx, :]
        n_good = np.sum(calib_metrics)
        if n_good > 0:
            calib_runs_with_nonzero_good[calib_idx] = calib_metrics

    good_images = dict()

    for k, v in calib_runs_with_nonzero_good.items():

        nonzero_indices = np.nonzero(v)[0]

        for idx in nonzero_indices:
            if idx not in good_images:
                good_images[idx] = 1
            else:
                good_images[idx] += 1

    good_counts = dict()

    for im_idx, c in good_images.items():

        if c not in good_counts:
            good_counts[c] = [im_idx]
        else:
            good_counts[c].append(im_idx)

    return calib_runs_with_nonzero_good, good_counts


def select_indices(good_counts, low_threshold=0):

    selected = []

    counts = filter(lambda c: c > low_threshold, good_counts.keys())

    for c in counts:
        selected += good_counts[c]

    return selected