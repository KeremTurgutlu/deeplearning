import os
import numpy as np
import cv2
from PIL import Image
import itertools


def list_files(path):
    """
    List files under a path
    """
    if path[-1] != '/':
        path = path + '/'
    return [path + p for p in os.listdir(path)]

def list_directory(path):
    """
    List directory under a path
    """
    if path[-1] != '/':
        path = path + '/'
    return [path + p + '/' for p in os.listdir(path)]

def get_image_onemask_paths(path, onemask=True):
    image_paths = [p + 'images/' + p.split('/')[-2] + '.png' for p in list_directory(path)]
    if onemask:
        mask_paths =  [p + 'one_mask.png' for p in list_directory(path)]
        return list(zip(image_paths, mask_paths))
    else:
        return image_paths

def get_img_mask_paths(MAIN_PATH, i):
    """
    return image file path and masks file path for i th image from sorted image folders list
    as well as the id of that img
    """
    IMG_FOLDER_PATHS = [MAIN_PATH + p for p in os.listdir(MAIN_PATH)]
    IMG_PATH = IMG_FOLDER_PATHS[i] + '/images/'
    MASK_PATHS = IMG_FOLDER_PATHS[i] + '/masks/'
    IMG_FILE_PATH = IMG_PATH + os.listdir(IMG_PATH)[0]
    MASK_FILES_PATH = [MASK_PATHS + p for p in os.listdir(MASK_PATHS)]

    return IMG_FILE_PATH, MASK_FILES_PATH, IMG_FOLDER_PATHS[i]


def create_one_mask_file(MASK_FILES_PATH, IMG_FOLDER_PATH):
    """
    This function will combine mask files into a single array
    Inputs:
        path: path of the masks directory
        target: where to save the new one mask
    """
    mask_arrays = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in MASK_FILES_PATH]
    zeros = np.zeros((mask_arrays[0].shape[0], mask_arrays[0].shape[1]))
    for arr in mask_arrays:
        zeros = zeros + arr
    zeros = np.dstack([zeros]*3).astype('uint8')
    im = Image.fromarray(zeros)
    im.save(IMG_FOLDER_PATH + '/one_mask.png')


def create_one_mask_arr(MASK_FILES_PATH):
    """
    This function will combine mask files into a single array
    Inputs:
        path: path of the masks directory
        target: where to save the new one mask
    """
    mask_arrays = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in MASK_FILES_PATH]
    zeros = np.zeros((mask_arrays[0].shape[0], mask_arrays[0].shape[1]))
    for arr in mask_arrays:
        zeros = zeros + arr

    return zeros



# read 2d - single channel image
def read2d(path): return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# read 3d - single channel image
def read3d(path): return cv2.imread(path, cv2.IMREAD_COLOR)

#############################
##    MULTICLASS MASK      ##
#############################

def get_contour_pos(mask):
    """
    Given a binary 2d mask this function will find contour positions of a mask

    Inputs:
        mask (np.array): binary 2d numpy array
    Returns:
        contour_pos (list): list which has (i, j) positions for contour pixels as elements
    """
    if mask.max() != 1:
        mask = mask / mask.max()
    H, W = mask.shape
    contour_pos = []
    for i in range(H):
        for j in range(W):
            # not edge
            m = []
            # N
            if (i) > 0:
                m.append(mask[i - 1, j])
            # S
            if (i) < H - 1:
                m.append(mask[i + 1, j])
            # E
            if (j) < W - 1:
                m.append(mask[i, j + 1])
            # W
            if (j) > 0:
                m.append(mask[i, j - 1])
            if (min(m) == 0) and (mask[i, j] == 1):
                contour_pos.append((i, j))
    return contour_pos

def get_contour_arr(mask):
    """
    This function will take a binary mask array and will return the contour of the mask
    Inputs:
        mask (np.array): binary 2d numpy array
    Returns:
        contour (np.array): binary 2d numpy array
    """
    if mask.max() != 1:
        mask_copy = mask.copy() / mask.max()
    mask_contour = mask_copy.copy()
    for i, j in get_contour_pos(mask_contour):
        mask_contour[i, j] = 2

    return (mask_contour - mask_copy)


def expand_contour(contour):
    """
    This function will expand a contour in a given set of directions by turning pixels into 1s
    Inputs:
        contour (np.array): Binary 2d numpy array which is the binary contour array
        directions (list): A list of tuples which has the movement addiiton in ith and jth position in tuples, e.g
        for moving left and right [(0, 1), (0, -1)], default is all 8 directions
    Returns:
        expanded_contour (np.array): Binary 2d numpy array which is the expanded contour array
    """
    all_directions = set(itertools.permutations([1, 1, -1, -1, 0, 0], 2))

    expanded_contour = contour.copy()
    for i, j in zip(np.where(contour == 1)[0], np.where(contour == 1)[1]):
        for i_plus, j_plus in all_directions:
            try:
                expanded_contour[i + i_plus, j + j_plus] = 1
            except:
                continue  # this is for the edges for now, not best practice but practical for fast solution
    return expanded_contour

def multiclass_onemask(sample_masks):
    """
    This function will return a multiclass mask having the labels:
        0: background
        1: foreground
        2: overlap boundary - 2 nuclei sticked together

    Inputs:
        sample_masks (list): a list of path of individual masks belonging to the same image
    """
    # get mask images
    mask_images = [read2d(mpath) for mpath in sample_masks]
    # get contour images
    contour_images = [get_contour_arr(mask) for mask in mask_images]
    # create expanded contour images
    expanded_contour_images = [expand_contour(contour) for contour in contour_images]
    # sum and assign to expanded contour images
    sum_expanded_contours = 0
    for expanded_contour in expanded_contour_images:
        sum_expanded_contours += expanded_contour
    sum_expanded_contours[sum_expanded_contours > 1] = 2
    # sum mask images and draw overlap boundaries
    sum_mask_images = 0
    for mask_image in mask_images:
        sum_mask_images += mask_image / 255  # make it binary
    sum_mask_images[sum_expanded_contours == 2] = 2

    return sum_mask_images





















