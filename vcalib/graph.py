import cv2
from epypes import compgraph
from visionfuncs import cbcalib
from visionfuncs import geometry


def create_stereo_cg():

    cg_base = CGCalibrateStereoBase()

    func_dict = {
        'prepare_corners': cbcalib.prepare_corners_stereo,
        'prepare_object_points': cbcalib.prepare_object_points,
    }

    func_io = {

        'prepare_corners': (
            ('calibration_images_1', 'calibration_images_2', 'pattern_size_wh'),
            ('image_points_1', 'image_points_2', 'num_images')
        ),

        'prepare_object_points': (
            ('num_images', 'pattern_size_wh', 'square_size'),
            'object_points'
        ),

    }

    cg_front = compgraph.CompGraph(func_dict, func_io)

    return compgraph.graph_union(cg_front, cg_base)


class CGCalibrateCamera(compgraph.CompGraph):

    def __init__(self):

        func_dict = {
            'prepare_corners': cbcalib.prepare_corners,
            'count_images': lambda lst: len(lst),
            'prepare_object_points': cbcalib.prepare_object_points,
            'calibrate_camera': cbcalib.calibrate_camera
        }

        func_io = {
            'prepare_corners': (('calibration_images', 'pattern_size_wh'), 'image_points'),
            'count_images': ('calibration_images', 'num_images'),
            'prepare_object_points': (('num_images', 'pattern_size_wh', 'square_size'), 'object_points'),
            'calibrate_camera': (('im_wh', 'object_points', 'image_points'),
                                 ('rms', 'camera_matrix', 'dist_coefs', 'rvecs', 'tvecs'))
        }

        super(CGCalibrateCamera, self).__init__(func_dict, func_io)


class CGSolvePnP(compgraph.CompGraph):

    def __init__(self):

        func_dict = {
            'detect_corners': cbcalib.find_corners_in_one_image,
            'solve_pnp': cv2.solvePnP,
            'rvec_to_rmat': geometry.rvec_to_rmat
        }

        func_io = {
            'detect_corners': (('image', 'pattern_size_wh'), 'image_points'),
            'solve_pnp': (('pattern_points', 'image_points', 'cam_matrix', 'dist_coefs'),
                          ('pnp_retval', 'rvec', 'tvec')),

            'rvec_to_rmat': ('rvec', 'rmat')
        }

        super(CGSolvePnP, self).__init__(func_dict, func_io)


class CGPreparePointsStereo(compgraph.CompGraph):

    def __init__(self):

        func_dict = {
            'prepare_corners_1': cbcalib.prepare_corners,
            'prepare_corners_2': cbcalib.prepare_corners,
            'prepare_indices': cbcalib.prepare_indices_stereocalib,
            'get_pattern_points': cbcalib.get_pattern_points,
        }

        func_io = {
            'prepare_corners_1': (('calibration_images_1', 'pattern_size_wh'), 'image_points_1'),
            'prepare_corners_2': (('calibration_images_2', 'pattern_size_wh'), 'image_points_2'),
            'prepare_indices': (('image_points_1', 'image_points_2'), 'indices'),
            'get_pattern_points': (('pattern_size_wh', 'square_size'), 'pattern_points'),
        }

        super(CGPreparePointsStereo, self).__init__(func_dict, func_io)


class CGCalibrateStereo(compgraph.CompGraph):

    def __init__(self):
        cg = create_stereo_cg()
        super(CGCalibrateStereo, self).__init__(cg.functions, cg.func_io)


class CGCalibrateStereoBase(compgraph.CompGraph):

    def __init__(self):

        func_dict = {
            'calibrate_camera_1': cbcalib.calibrate_camera,
            'calibrate_camera_2': cbcalib.calibrate_camera,
            'calibrate_stereo': cbcalib.calibrate_stereo,
            'compute_rectification_transforms': cv2.stereoRectify
        }

        func_io = {
            'calibrate_camera_1': (('im_wh', 'object_points', 'image_points_1'),
                                   ('rms_1', 'cm_1', 'dc_1', 'rvecs_1', 'tvecs_1')),
            'calibrate_camera_2': (('im_wh', 'object_points', 'image_points_2'),
                                   ('rms_2', 'cm_2', 'dc_2', 'rvecs_2', 'tvecs_2')),
            'calibrate_stereo': (('object_points', 'image_points_1', 'image_points_2', 'cm_1', 'dc_1', 'cm_2', 'dc_2', 'im_wh'),
                                 ('stereo_rmat', 'stereo_tvec', 'essential_mat', 'fundamental_mat')),
            'compute_rectification_transforms': (('cm_1', 'dc_1', 'cm_2', 'dc_2', 'im_wh', 'stereo_rmat', 'stereo_tvec'),
                                                 ('R1', 'R2', 'P1', 'P2', 'Q', 'validPixROI1', 'validPixROI2'))
        }

        super(CGCalibrateStereoBase, self).__init__(func_dict, func_io)


class CGFindCorners(compgraph.CompGraph):

     def __init__(self):

         func_dict = {
             'find_corners': cbcalib.find_cbc,
             'reformat_corners': cbcalib.cbc_opencv_to_numpy
         }

         func_io = {
             'find_corners': (('image', 'pattern_size_wh'), ('success', 'corners_opencv')),
             'reformat_corners': (('success', 'corners_opencv'), 'corners_np')
         }

         super(CGFindCorners, self).__init__(func_dict, func_io)
