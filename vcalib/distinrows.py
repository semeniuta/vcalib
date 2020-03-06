import numpy as np


def cb_row_by_row(pattern_size):
    """
    Generator of pairs of (start, end)
    indices of a chessboard pattern 
    per each row. 
    """

    n_cols, n_rows = pattern_size

    idx = 0

    for i in range(n_rows):
        start = i * n_cols
        end = start + n_cols
        yield start, end


def measure_cb_distances_in_rows(points_3d, pattern_size):

    distances = []

    n_cols, n_rows = pattern_size
    assert len(points_3d) == (n_cols * n_rows)

    for start, end in cb_row_by_row(pattern_size):

        row_points = points_3d[start:end]

        for i in range(len(row_points) - 1):
            p1 = row_points[i]
            p2 = row_points[i + 1]
            dist = np.linalg.norm(p1 - p2)
            distances.append(dist)

    return np.array(distances)
