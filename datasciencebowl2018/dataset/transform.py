import cv2
import random
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


######################################
####      2D TRANSFORMS          ####
#####################################


# resize
def fix_resize_transform2(image, mask, w, h):
    image = cv2.resize(image, (w, h))
    mask = cv2.resize(mask, (w, h))
    mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
    return image, mask

def fix_resize_transform(image, w, h):
    image = cv2.resize(image, (w, h))
    return image

# horizontal flip
def random_horizontal_flip_transform2(image, mask, p=0.5):
    if random.random() < p:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask

def random_horizontal_flip_transform(image, p=0.5):
    if random.random() < p:
        image = cv2.flip(image, 1)
    return image


# vertical flip
def random_vertical_flip_transform2(image, mask, p=0.5):
    if random.random() < p:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    return image, mask

def random_vertical_flip_transform(image, p=0.5):
    if random.random() < p:
        image = cv2.flip(image, 0)
    return image


# 90, 180, 270 rotate
def random_rotate_transform2(image, mask, p=0.5):
    H, W = mask.shape[0], mask.shape[1]
    if random.random() < p:
        angle = random.randint(1, 3)*90
        M = cv2.getRotationMatrix2D((H/2, W/2), angle, 1)
        image = cv2.warpAffine(image, M, (H, W))
        mask = cv2.warpAffine(mask, M, (H, W))
    return image, mask

def random_rotate_transform(image, p=0.5):
    H, W = image.shape[0], image.shape[1]
    if random.random() < p:
        angle = random.randint(1, 3)*90
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
        mask = cv2.threshold(mask, 128, 215, cv2.THRESH_BINARY)[1]
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

        image = cv2.resize(image, (W, H))
        mask = cv2.resize(mask, (W, H))
        mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
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
    return image



######################################
####      3D TRANSFORMS          ####
#####################################

def fix_resize_3D_transform2(image_voxel, mask_voxel, w, h):
    image_voxel = np.array([cv2.resize(slice_, (w, h)) for slice_ in image_voxel])
    mask_voxel = np.array([cv2.resize(slice_, (w, h)) for slice_ in mask_voxel])
    # assign 215 to every pixel greater than 128
    mask_voxel = cv2.threshold(mask_voxel.astype(np.uint8), 128, 215, cv2.THRESH_BINARY)[1]
    return image_voxel, mask_voxel

def fix_resize_3D_transform(image_voxel, w, h):
    image_voxel = np.array([cv2.resize(slice_, (w, h)) for slice_ in image])
    return image_voxel

# horizontal flip 3D
def random_horizontal_flip_3D_transform2(image_voxel, mask_voxel, p=0.5):
    if random.random() < p:
        image_voxel = np.array([cv2.flip(slice_, 1) for slice_ in image_voxel])
        mask_voxel = np.array([cv2.flip(slice_, 1) for slice_ in mask_voxel])
    return image_voxel, mask_voxel

def random_horizontal_flip3D_transform(image_voxel, p=0.5):
    if random.random() < p:
        image_voxel = np.array([cv2.flip(slice_, 1) for slice_ in image_voxel])
    return image_voxel


# vertical flip 3D
def random_vertical_flip_3D_transform2(image_voxel, mask_voxel, p=0.5):
    if random.random() < p:
        image_voxel = np.array([cv2.flip(slice_, 0) for slice_ in image_voxel])
        mask_voxel = np.array([cv2.flip(slice_, 0) for slice_ in mask_voxel])
    return image_voxel, mask_voxel

def random_vertical_flip_3D_transform(image_voxel, p=0.5):
    if random.random() < p:
        image_voxel = np.array([cv2.flip(slice_, 0) for slice_ in image_voxel])
    return image_voxel


# 90, 180, 270 rotate 3D
def random_rotate_3D_transform2(image_voxel, mask_voxel, p=0.5):
    H, W = mask_voxel.shape[-1], mask_voxel.shape[-2]
    if random.random() < p:
        angle = random.randint(1, 3) * 90
        M = cv2.getRotationMatrix2D((H / 2, W / 2), angle, 1)
        image_voxel = np.array([cv2.warpAffine(slice_, M, (H, W)) for slice_ in image_voxel])
        mask_voxel = np.array([cv2.warpAffine(slice_, M, (H, W)) for slice_ in mask_voxel])
    return image_voxel, mask_voxel

def random_rotate_3D_transform(image_voxel, p=0.5):
    H, W = image_voxel.shape[-1], image_voxel.shape[-2]
    if random.random() < p:
        angle = random.randint(1, 3) * 90
        M = cv2.getRotationMatrix2D((H / 2, W / 2), angle, 1)
        image_voxel = np.array([cv2.warpAffine(slice_, M, (H, W)) for slice_ in image_voxel])
    return image_voxel


# range(0, 360, 10) angle rotate
def random_rotate_angle_3D_transform2(image_voxel, mask_voxel, p=0.5):
    angle = np.random.choice(range(0, 360, 10))
    H, W = image_voxel.shape[-1], image_voxel.shape[-2]
    if random.random() < p:
        M = cv2.getRotationMatrix2D((H/2, W/2), angle, 1)
        image_voxel = np.array([cv2.warpAffine(slice_, M, (H, W)) for slice_ in image_voxel])
        mask_voxel = np.array([cv2.warpAffine(slice_, M, (H, W)) for slice_ in mask_voxel])
    mask_voxel = np.array([cv2.threshold(slice_, 128, 215, cv2.THRESH_BINARY)[1] for slice_ in mask_voxel])
    return image_voxel, mask_voxel

def random_rotate_angle_3D_transform(image_voxel, p=0.5):
    angle = np.random.choice(range(0, 360, 10))
    H, W = image_voxel.shape[-1], image_voxel.shape[-2]
    if random.random() < p:
        M = cv2.getRotationMatrix2D((H/2, W/2), angle, 1)
        image_voxel = np.array([cv2.warpAffine(slice_, M, (H, W)) for slice_ in image_voxel])
    return image_voxel

# RANDOM crop resize

def random_crop_3D_resize2(image_voxel, mask_voxel, p=0.5):
    H, W = image_voxel.shape[-1], image_voxel.shape[-2]
    if random.random() < p:
        # PICK A RANDOM POINT
        j, i = np.random.choice(range(H)), np.random.choice(range(W))

        # UPPER LEFT
        if (j <= H / 2) & (i <= W / 2):
            image_voxel = np.array([slice_[j:H, i:H] for slice_ in image_voxel])
            mask_voxel = np.array([slice_[j:H, i:H] for slice_ in mask_voxel])

        # LOWER LEFT
        elif (j >= H / 2) & (i <= W / 2):
            image_voxel = np.array([slice_[:j, i:H] for slice_ in image_voxel])
            mask_voxel = np.array([slice_[:j, i:H] for slice_ in mask_voxel])

        # UPPER RIGHT
        elif (j <= H / 2) & (i >= W / 2):
            image_voxel = np.array([slice_[j:H, :i] for slice_ in image_voxel])
            mask_voxel = np.array([slice_[j:H, :i] for slice_ in mask_voxel])

        # LOWER RIGHT
        elif (j >= H / 2) & (i >= W / 2):
            image_voxel = np.array([slice_[:j, :i] for slice_ in image_voxel])
            mask_voxel = np.array([slice_[:j, :i] for slice_ in mask_voxel])

        else:
            image_voxel, mask_voxel = image_voxel, mask_voxel

        image_voxel = np.array([cv2.resize(slice_, (W, H)) for slice_ in image_voxel])
        mask_voxel = np.array([cv2.resize(slice_, (W, H)) for slice_ in mask_voxel])
        mask_voxel = np.array([cv2.threshold(slice_, 128, 215, cv2.THRESH_BINARY)[1] for slice_ in mask_voxel])
    return image_voxel, mask_voxel


def random_crop_3D_resize(image_voxel, p=0.5):
    H, W = image_voxel.shape[-1], image_voxel.shape[-2]
    if random.random() < p:
        # PICK A RANDOM POINT
        j, i = np.random.choice(range(H)), np.random.choice(range(W))

        # UPPER LEFT
        if (j <= H / 2) & (i <= W / 2):
            image_voxel = np.array([slice_[j:H, i:H] for slice_ in image_voxel])
            # LOWER LEFT
        elif (j >= H / 2) & (i <= W / 2):
            image_voxel = np.array([slice_[:j, i:H] for slice_ in image_voxel])

        # UPPER RIGHT
        elif (j <= H / 2) & (i >= W / 2):
            image_voxel = np.array([slice_[j:H, :i] for slice_ in image_voxel])

        # LOWER RIGHT
        elif (j >= H / 2) & (i >= W / 2):
            image_voxel = np.array([slice_[:j, :i] for slice_ in image_voxel])

        else:
            image_voxel = image_voxel

        image_voxel = np.array([cv2.resize(slice_, (W, H)) for slice_ in image_voxel])
    return image_voxel


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(random_state)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def random_elastic_3D_transform2(image_voxel, mask_voxel, p=0.5):
    if random.random() < p:
        random_state = np.random.randint(1)
        new_image_voxel, new_mask_voxel = [], []
        # Merge images into separete channels (shape will be (cols, rols, 2))
        for im, im_mask in zip(image_voxel, mask_voxel):
            im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)
            im_merge_t = elastic_transform(image=im_merge,
                                           alpha=im_merge.shape[1] * 2,
                                           sigma=im_merge.shape[1] * 0.1,
                                           alpha_affine=im_merge.shape[1] * 0.01,
                                           random_state=random_state)
            # Split image and mask
            im_t = im_merge_t[..., 0]
            im_mask_t = im_merge_t[..., 1] > 0
            new_image_voxel.append(im_t)
            new_mask_voxel.append(im_mask_t)
        image_voxel, mask_voxel = np.array(new_image_voxel), np.array(new_mask_voxel)
    return image_voxel, mask_voxel
