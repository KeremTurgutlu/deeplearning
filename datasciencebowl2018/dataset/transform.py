import cv2
import random
import numpy as np

# Note on TTA
# IN ORDER TO APPLY TTA WE NEED TO KNOW WHICH TRANSFORMS
# WE ARE DOING ON THE TEST DATA SO THAT WE CAN RE-NORMALIZE IT
# ANOTHER NOTE IS THAT WE NEED TO RESIZE THE LABELS BACK

# resize
def fix_resize_transform2(image, mask, w, h):
    image = cv2.resize(image,(w,h))
    mask  = cv2.resize(mask,(w,h))
    mask  = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
    return image, mask

def fix_resize_transform(image, w, h):
    image = cv2.resize(image,(w,h))
    return image

# horizontal flip
def random_horizontal_flip_transform2(image, mask, p=0.5):
    if random.random() < p:
        image = cv2.flip(image,1)
        mask = cv2.flip(mask,1)
    return image, mask

def random_horizontal_flip_transform(image, p=0.5):
    if random.random() < p:
        image = cv2.flip(image,1)
    return image


# vertical flip
def random_vertical_flip_transform2(image, mask, p=0.5):
    if random.random() < p:
        image = cv2.flip(image,0)
        mask = cv2.flip(mask,0)
    return image, mask

def random_vertical_flip_transform(image, p=0.5):
    if random.random() < p:
        image = cv2.flip(image,0)
    return image


# 90, 180, 270 rotate
def random_rotate_transform2(image, mask, p=0.5):
    H, W = mask.shape[0], mask.shape[1]
    if random.random() < p:
        angle=random.randint(1,3)*90
        M = cv2.getRotationMatrix2D((H/2, W/2), angle, 1)
        image = cv2.warpAffine(image, M, (H, W))
        mask = cv2.warpAffine(mask, M, (H, W))
    return image, mask

def random_rotate_transform(image, p=0.5):
    H, W = image.shape[0], image.shape[1]
    if random.random() < p:
        angle=random.randint(1,3)*90
        M = cv2.getRotationMatrix2D((H/2, W/2), angle, 1)
        image = cv2.warpAffine(image, M, (H, W))
    return image

# range(0, 360, 10) angle rotate
def random_rotate_angle_transform2(image, mask, p=0.5):
    angle = np.random.choice(range(0, 360, 10))
    H, W = image.shape[0], image.shape[1]
    if random.random() < p:
        M = cv2.getRotationMatrix2D((H/2, W/2), angle, 1)
        image = cv2.warpAffine(image, M, (H, W))
        mask = cv2.warpAffine(mask, M, (H, W))
    return image, mask

def random_rotate_angle_transform(image, p=0.5):
    angle = np.random.choice(range(0, 360, 10))
    H, W = image.shape[0], image.shape[1]
    if random.random() < p:
        M = cv2.getRotationMatrix2D((H/2, W/2), angle, 1)
        image = cv2.warpAffine(image, M, (H, W))
    return image

# random crop resize
# it's better to do this augmentation before
# resizing for net
def random_crop_resize2(image, mask, p=0.5):
    H, W, _ = image.shape
    if random.random() < p:
        # PICK A RANDOM POINT
        j, i = np.random.choice(range(H)), np.random.choice(range(W))

        # UPPER LEFT
        if (j <= H/2) & (i <= W/2):
            image, mask = image[j:H, i:H], mask[j:H, i:H]
        # LOWER LEFT
        elif (j >= H/2) & (i <= W/2):
            image, mask = image[:j, i:H], mask[:j, i:H]
        # UPPER RIGHT
        elif (j <= H/2) & (i >= W/2):
            image, mask = image[j:H, :i], mask[j:H, :i]
        # LOWER RIGHT
        elif (j >= H/2) & (i >= W/2):
            image, mask = image[:j, :i], mask[:j, :i]
        else:
            image, mask = image, mask

        image = cv2.resize(image,(W,H))
        mask  = cv2.resize(mask,(W,H))
        mask  = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
    return image, mask


def random_crop_resize(image, p=0.5):
    H, W, _ = image.shape
    if random.random() < p:
        # PICK A RANDOM POINT
        j, i = np.random.choice(range(H)), np.random.choice(range(W))

        # UPPER LEFT
        if (j <= H / 2) & (i <= W / 2):
            image = image[j:H, i:H]
        # LOWER LEFT
        elif (j >= H / 2) & (i <= W / 2):
            image = image[:j, i:H]
        # UPPER RIGHT
        elif (j <= H / 2) & (i >= W / 2):
            image = image[j:H, :i]
        # LOWER RIGHT
        elif (j >= H / 2) & (i >= W / 2):
            image = image[:j, :i]
        else:
            image = image

        image = cv2.resize(image, (W, H))
        mask = cv2.resize(mask, (W, H))
        mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
    return image, mask


# transforming 2d mask to multichannel
def get_3d_mask(mask):
    if len(np.unique(mask)) == 2:
        back_channel = (mask == 30)*1 # background
        nuclei_channel = (mask == 215)*1 # nuclei
        overlap_channel = np.zeros_like(mask) # overlap - missing so all 0s
    else:
        back_channel = (mask == 30)*1 # background
        nuclei_channel = (mask == 110)*1 # nuclei
        overlap_channel = (mask == 215)*1 # overlap
    # stack depth-wise
    multiclass_mask = np.dstack([back_channel, nuclei_channel, overlap_channel])
    return multiclass_mask



# ELASTIC TRANSFORMATION

# import numpy as np
# from scipy.ndimage.interpolation import map_coordinates
# from scipy.ndimage.filters import gaussian_filter
#
# def elastic_transform(image, alpha, sigma, random_state=None):
#     """Elastic deformation of images as described in [Simard2003]_.
#     .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
#        Convolutional Neural Networks applied to Visual Document Analysis", in
#        Proc. of the International Conference on Document Analysis and
#        Recognition, 2003.
#     """
#     if random_state is None:
#         random_state = np.random.RandomState(None)
#
#     shape = image.shape
#     dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
#     dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
#     dz = np.zeros_like(dx)
#
#     x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
#     indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
#
#     distored_image = map_coordinates(image, indices, order=1, mode='reflect')
#     return distored_image.reshape(image.shape)
#
#
#
#
#
#
#






