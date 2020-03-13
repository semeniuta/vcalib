import numpy as np


def move_rows_around(mat, row_seq):

    res = [mat[row] for row in row_seq]

    return np.array(res)


def make_mask_from_indices(total_len, indices):

    mask = np.zeros(total_len, dtype=bool)
    for i in indices:
        mask[i] = True

    return mask


def make_opposite_masks_from_indices(total_len, indices):

    mask = make_mask_from_indices(total_len, indices)
    mask_not = ~mask

    return mask, mask_not
