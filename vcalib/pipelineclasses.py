from visioncg import cbcalib
from epypes.compgraph import CompGraphRunner
from .imsubsets import shuffle
from .calibrun import prepare_points_for_all_images
from .calibrun import triangulate_all
from .calibrun import run_calib_for_subsets
from .calibim import create_metric_mean_dist_in_rows
from .calibim import apply_metric_to_all_point_clouds
from .calibim import detect_good_triangulations
from .calibim import summarize_good_vals
from .calibim import get_good_vals
from .calibim import create_good_vals_histograms
from .calibim import augment_df_good_vals_with_hist


class CalibrationInput:

    def __init__(self, imfiles_1, imfiles_2, psize, sqsize):

        self.pattern_size = psize
        self.square_size = sqsize

        cg = cbcalib.CGPreparePointsStereo()
        params = {'pattern_size_wh': psize, 'square_size': sqsize}

        self.runner_prepare = CompGraphRunner(cg, params)
        prepare_points_for_all_images(self.runner_prepare, imfiles_1, imfiles_2)

        self.indices = self.runner_prepare['indices'].copy()
        self.n_images = len(self.indices)

        im0 = self.runner_prepare['calibration_images_1'][0]
        h, w = im0.shape
        self.im_wh = (w, h)


    def shuffle_indices(self, shuffle_seed=42):
        shuffle(self.indices, seed=shuffle_seed)


class CalibTriang:

    def __init__(self, calib_input, subsets):

        self.calib_input = calib_input
        self.calib_runners = run_calib_for_subsets(subsets, calib_input.runner_prepare, calib_input.im_wh)
        self.triang = triangulate_all(self.calib_runners, calib_input.runner_prepare)


class MeanDistInRows: 

    def __init__(self, calib_triang):

        ps = calib_triang.calib_input.pattern_size
        metric_func = create_metric_mean_dist_in_rows(psize=ps)

        self.metric_mat = apply_metric_to_all_point_clouds(calib_triang.triang, metric_func)

        
class ValuesAroundTarget:

    def __init__(self, metric_mat, target, tol):

        self.mask = detect_good_triangulations(metric_mat, target, tol)
        
        good_vals = get_good_vals(metric_mat, self.mask)
        good_vals_df = summarize_good_vals(good_vals, nominal_value=target)
        good_vals_hist = create_good_vals_histograms(good_vals, nominal_value=target)
        
        self.df = augment_df_good_vals_with_hist(good_vals_df, good_vals_hist)

    def df_sorted(self):
        return self.df.sort_values(['Hist0', 'MaxAbsErr'], ascending=[False, True])


class MaskedValues:

    def __init__(self, metric_mat, target, mask):

        self.mask = mask
        
        good_vals = get_good_vals(metric_mat, self.mask)
        good_vals_df = summarize_good_vals(good_vals, nominal_value=target)
        good_vals_hist = create_good_vals_histograms(good_vals, nominal_value=target)
        
        self.df = augment_df_good_vals_with_hist(good_vals_df, good_vals_hist)

    def df_sorted(self):
        return self.df.sort_values(['Hist0', 'MaxAbsErr'], ascending=[False, True])
