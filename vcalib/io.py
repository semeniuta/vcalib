import cv2

from visionfuncs.io import open_image


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