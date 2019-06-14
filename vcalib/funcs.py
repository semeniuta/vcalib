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