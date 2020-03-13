import numpy as np
import random
from math import factorial
from .nputils import make_mask_from_indices


def shuffle(x, seed=None):
    
    if seed is None:
        rnd = random.Random() 
    else:
        rnd = random.Random(seed) 
        
    rnd.shuffle(x)
    
    
def shuffle_indices(n, seed=42):
    
    indices = list(range(n))
    shuffle(indices, seed)
    
    return indices


def n_choose_k(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))


def subsets_sliding(indices, subset_size):
    
    total = len(indices)
    
    if subset_size > total:
        raise Exception('subset_size should be less or equal to the size of indices')
        
    last = total - subset_size
        
    for start in range(0, last + 1):
        yield indices[start:start+subset_size]


def sample_subsets(elements, subset_size, n_subsets, seed):

    np.random.seed(seed)
    seeds = np.random.randint(0, 100 * n_subsets, n_subsets)

    for s in seeds:
        np.random.seed(s)
        subset = np.random.choice(elements, subset_size, replace=False)

        yield subset


def subsets_onehot(subsets, n_images):

    res = np.zeros((len(subsets), n_images))

    for i, ss in enumerate(subsets):
       res[i, :] = make_mask_from_indices(n_images, ss)

    return res


def gather_image_indices_in_calib_runs(subsets, calib_indices):

    res = set()

    for i in calib_indices:
        res.update(subsets[i])

    return np.array(sorted(res))


def gather_image_frequencies_in_calib_runs(subsets, calib_indices, n_images):

    freq = np.zeros(n_images, dtype=int)

    for i in calib_indices:

        for im_idx in subsets[i]:
            freq[im_idx] += 1

    return freq
