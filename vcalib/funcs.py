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

        
def open_images_all(imfiles):
    
    def open_image_gray(im_filename):
        return open_image(im_filename, read_flag=cv2.IMREAD_GRAYSCALE)
    
    return [open_image_gray(imf) for imf in imfiles] 
    
      
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


def prepare_points_for_all_images(runner_prepare, imfiles_1, imfiles_2):
    
    all_images_1 = open_images_all(imfiles_1)
    all_images_2 = open_images_all(imfiles_2)
    
    runner_prepare.run(
        calibration_images_1=all_images_1,
        calibration_images_2=all_images_2
    )
    
    
def run_calib(runner_calib, impoints_1, impoints_2, indices_subset, pattern_points):
    
    imp_1 = [impoints_1[idx] for idx in indices_subset]
    imp_2 = [impoints_2[idx] for idx in indices_subset]
    obp = cbcalib.make_list_of_identical_pattern_points(len(indices_subset), pattern_points)

    runner_calib.run(
        image_points_1=imp_1,
        image_points_2=imp_2,
        object_points=obp
    )
        
    
def all_images_reprojection_error_for_subsets(indices_subset_gen, runner_prepare, runner_calib):
    """
    For each indices subset generated by indices_subset_gen,
    perform stereo calibration. Then, given the resulting intrinsics,
    solve PnP problem and compute reprojection error for all images.
    Return two NumPy arrays of equal length, where each element 
    corresponds to reprojection error given all images 
    and intrinsics from calibration based on a specific images subset.
    """
    
    rms_list_1 = []
    rms_list_2 = []
    
    # for all images
    impoints_1 = runner_prepare['image_points_1']
    impoints_2 = runner_prepare['image_points_2']
    pattern_points = runner_prepare['pattern_points']
    
    def multiple_pnp(impoints, cm, dc): # capturing object_points

        rvecs = []
        tvecs = []

        for imp in impoints:
            _, rvec, tvec = cv2.solvePnP(pattern_points, imp, cm, dc) 
            rvecs.append(rvec)
            tvecs.append(tvec)

        return rvecs, tvecs
    
    object_points = cbcalib.make_list_of_identical_pattern_points(len(impoints_1), pattern_points)

    for indices_subset in indices_subset_gen:
       
        run_calib(runner_calib, impoints_1, impoints_2, indices_subset, pattern_points)
        
        cm1 = runner_calib['cm_1']
        dc1 = runner_calib['dc_1']
        cm2 = runner_calib['cm_2']
        dc2 = runner_calib['dc_2']
        
        rvecs1, tvecs1 = multiple_pnp(impoints_1, cm1, dc1)
        rvecs2, tvecs2 = multiple_pnp(impoints_2, cm2, dc2)
        
        rms1 = reproject_and_measure_error(impoints_1, object_points, rvecs1, tvecs1, cm1, dc1)
        rms2 = reproject_and_measure_error(impoints_2, object_points, rvecs2, tvecs2, cm2, dc2)

        rms_list_1.append(rms1)
        rms_list_2.append(rms2)
        
        """
        p3d = geometry.triangulate_points(
            runner_calib['P1'],
            runner_calib['P2'],
            runner_prepare['image_points_1'][0],
            runner_prepare['image_points_2'][0]
        )
        print(np.linalg.norm(p3d[0] - p3d[1]))
        """
        
    return np.array(rms_list_1), np.array(rms_list_2)