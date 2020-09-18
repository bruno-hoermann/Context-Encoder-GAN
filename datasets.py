
import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.jpg" % root))
        self.files = self.files[:-4000] if mode == "train" else self.files[-4000:]

    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, i

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == "train":
            # For training data perform random mask
            masked_img, aux = self.apply_random_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux = self.apply_center_mask(img)

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)

class CelebADataset(torchvision.datasets.CelebA):
    """
    Specializes CelebA dataset class and updates the filelist ids to download from private google drive. The checksum
    is same so the files should work.
    """
    file_list = [
            # File ID                         MD5 Hash                            Filename
            ("1DV305oDyaBJ9fZG_IPlRVDtiDTyCQ7MG", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
            ("1n7QGLRUYunboJOQxxVapBWqaevJxtBFa", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
            ("1PwHL-ithk1JkHKkM-50_dDWUKpwCElyO", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
            ("1DrvSKABlfFP7E8Oj_lINnf6HOOS10sUZ", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
            ("1dVB-E6n5DekrrsyhnvwHRAJluaJFmar6", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
            ("1D19UoRHTbFbNZbe3pOP8JikWsjRkiUda", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
        ]
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
        