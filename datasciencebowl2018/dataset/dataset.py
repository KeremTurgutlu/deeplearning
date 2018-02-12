from utility.utils import list_directory
from torch.utils.data import *
import cv2


class NucleiDataset(Dataset):
    def __init__(self, path, transform=None, mode='train'):
        self.path = path
        self.transform = transform
        self.mode = mode
        self.image_dirs = list_directory(path)

    def __getitem__(self, index):
        # Get filenames
        image_dir = self.image_dirs[index]
        image_id = image_dir.split('/')[-2]
        image_file = image_dir + 'images/' + image_id + '.png'

        # Get mask and read files
        if self.mode in ['train', 'valid']:
            mask_file = image_dir + '/one_mask.png'
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if self.transform is not None:
                return self.transform(image, mask, image_id)
            else:
                return image, mask, image_id
        # Just read files
        else:
            if self.transform is not None:
                return self.transform(image, None, image_id)
            else:
                return image, image_id


    def __len__(self):
        return len(self.image_dirs)