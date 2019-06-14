import cv2
import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from math import factorial
import itertools
import random
from matplotlib import pyplot as plt

from visioncg.io import sorted_glob
from visioncg.io import open_image


def glob_images(data_dir, camera_idx):
    mask = os.path.join(data_dir, 'img_{}_*.jpg'.format(camera_idx))
    return sorted_glob(mask)


def n_choose_k(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))


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


def imshow_noaxis(ax, im):
    ax.imshow(im)
    ax.axis('off')
    
    
def subsets_sliding(indices, subset_size):
    
    total = len(indices)
    
    if subset_size > total:
        raise Exception('subset_size should be less or equal to the size of indices')
        
    last = total - subset_size
        
    for start in range(0, last + 1):
        yield indices[start:start+subset_size]
    
    
def open_images_subset(imfiles, subset_indices):
    
    def open_image_gray_by_idx(idx):
        return open_image(imfiles[idx], read_flag=cv2.IMREAD_GRAYSCALE)
    
    return [open_image_gray_by_idx(idx) for idx in subset_indices]


def open_images_subset_stereo(imfiles_1, imfiles_2, subset_indices):
    
    images_1 = open_images_subset(imfiles_1, subset_indices)
    images_2 = open_images_subset(imfiles_2, subset_indices)
    
    return images_1, images_2

    
def calibrate_stereo(images_1, images_2):
    
    cg = cbcalib.CGCalibrateStereo()

    params = {
        'im_wh': cbcalib.get_im_wh(images_1[0]),
        'pattern_size_wh': (9, 7),
        'square_size': 20.   
    }

    runner = CompGraphRunner(cg, params)
    
    runner.run(calibration_images_1=images_1, calibration_images_2=images_2)
    
    return runner


def reproject_and_measure_error(image_points, object_points, rvecs, tvecs, cm, dc):
    
    reproj_list = []
    
    for ip, op, rvec, tvec in zip(image_points, object_points, rvecs, tvecs):

        ip_reprojected = cbcalib.project_points(op, rvec, tvec, cm, dc)
        reproj_list.append(ip_reprojected)
        
    reproj_all = np.concatenate(reproj_list, axis=0)
    original_all = np.concatenate(image_points, axis=0)
    
    rms = cbcalib.reprojection_rms(original_all, reproj_all)
    return rms