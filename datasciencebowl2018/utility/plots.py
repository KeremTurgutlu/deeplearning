import numpy as np
import cv2
import matplotlib.pyplot as plt



def show_img_msk_frompath(img_path, msk_path=None, alpha=0.35, sz=7):
    """
    Show original image and masked on top of image
    next to each other in desired size
    Input:
        img_path (np.array): path of the image
        msk_path (np.array): path of the mask
        alpha (float): a number between 0 and 1 for mask transparency
        sz (int): figure size for display
    """
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    plt.figure(figsize=(sz, sz))
    if msk_path is not None:
        image_mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        masked_image = np.ma.masked_where(image_mask == 0, image_mask)
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.imshow(masked_image, cmap='cool', alpha=alpha)
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image)
        plt.show()

def show_img_msk_fromarray(img_arr, msk_arr, alpha=0.35, sz=7):

    """
    Show original image and masked on top of image
    next to each other in desired size
    Input:
        img_arr (np.array): array of the image
        msk_arr (np.array): array of the mask
        alpha (float): a number between 0 and 1 for mask transparency
        sz (int): figure size for display
    """

    msk_arr = np.ma.masked_where(msk_arr == 0, msk_arr)
    plt.figure(figsize=(sz, sz))
    plt.subplot(1, 2, 1)
    plt.imshow(img_arr)
    plt.imshow(msk_arr, cmap='cool', alpha=alpha)
    plt.subplot(1, 2, 2)
    plt.imshow(img_arr)
    plt.show()

def show_with_sz(img, sz=7):
    """
    Creates a single figure for single image and displays it with
    desired figrue size
    Inputs:
        image (np.array): array to display
        sz (int): figure size for display
    """
    plt.figure(figsize=(sz, sz))
    plt.imshow(img)
    plt.show()
    plt.close()

def show_side_to_side(img1, img2, sz=7):
    """
    Creates a single figure for two images to be displayed
    in desired size next to each other
    Inputs:
        img1 (np.array): array to display
        img2 (np.array): array to display
        sz (int): figure size for display
    """
    plt.figure(figsize=(sz, sz))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.show()
    plt.close()


def show_predictions(dataloader, classifier, threshold=0.5, n=None):
    print('\t\t Image \t\t\t\t\t Mask \t\t\t\t Predicted Mask')
    for i, (img, msk, _ )in enumerate(iter(dataloader)):
        classifier.net.eval()
        plt.figure(figsize=(20, 20))
        if torch.cuda.is_available():
            img = img.cuda()
        out = classifier.net(V(img))
        plt.subplot(1,3,1)
        plt.imshow(img.cpu().numpy()[0].transpose(1,2,0))
        plt.subplot(1,3,2)
        plt.imshow(msk.cpu().numpy()[0, 0])
        plt.subplot(1,3,3)
        plt.imshow((F.sigmoid(out).cpu().data.numpy()[0, 0] > threshold)*1)
        plt.show()
        i += 1
        if n:
            if n == i:
                break

# FASTAI
def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax