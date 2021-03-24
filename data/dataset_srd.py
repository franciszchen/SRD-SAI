import os
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def pil2tensor(image, dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False))

class WCE_Dataset_npy(Dataset):
    def __init__(self, cross_str, stage, folder='your dataset folder', size=128, augmentations=True):
        """
        random_crop=False,
        random_fliplr=True, random_flipud=True, random_rotation=True,
        color_jitter=False, brightness=0.1
        """
        super(WCE_Dataset_npy, self).__init__()
        # basic settings
        self.folder = folder
        self.size = size
        self.stage = stage
        self.augmentations = augmentations

        # color augmentation
        self.RANDOM_BRIGHTNESS = 7
        self.RANDOM_CONTRAST = 5
        # dataset load
        self._raw_ims = (np.load(os.path.join(folder, cross_str+'_new_shuffle_wce'+str(self.size)+'_'+self.stage+'_img.npy'),encoding="latin1")).item(0)
        self._raw_lbs = (np.load(os.path.join(folder, cross_str+'_new_shuffle_wce'+str(self.size)+'_'+self.stage+'_label.npy'),encoding="latin1")).item(0)
        print('load {} {:d} dataset'.format(self.stage, self.size))

    def __getitem__(self, index):
        im = self._raw_ims[str(index)]
        lb = int(self._raw_lbs[str(index)])
        # im = np.expand_dims(im, axis=2).copy()

        if self.augmentations:
            # random flip
            if random.uniform(0, 1) < 0.5:
                im = np.fliplr(im)
            if random.uniform(0, 1) < 0.5:
                im = np.flipud(im)

            # random rotation
            r = random.randint(0, 3)
            if r:
                im = np.rot90(im, r)
            # cast to float
            im = im.astype(np.float32) / 255.0 # [0, 1]
            # color jitter
            br = random.randint(-self.RANDOM_BRIGHTNESS, self.RANDOM_BRIGHTNESS) / 100.
            im = im + br
            # Random contrast
            cr = 1.0 + random.randint(-self.RANDOM_CONTRAST, self.RANDOM_CONTRAST) / 100.
            im = im * cr
            # clip values to 0-1 range
            im = np.clip(im, 0, 1.0)
            im = pil2tensor(im, np.float32)
        else:
            im = im.astype(np.float32) / 255.0
            im = pil2tensor(im, np.float32)
            # [0, 1] 
        return im, lb

    def __len__(self):
        return len(self._raw_lbs)


def get_loader(batch_size, cross_str, stage, size, num_workers, augmentations):

    if stage == 'train':
        dataset = WCE_Dataset_npy(cross_str=cross_str, stage='train', size=size, augmentations=augmentations)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        print('build WCE{:d} train dataset with {} num_workers'.format(size, num_workers))
    elif stage == 'test':
        dataset = WCE_Dataset_npy(cross_str=cross_str, stage='test', size=size, augmentations=augmentations)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
        print('build WCE{:d} test dataset with {} num_workers'.format(size, num_workers))
    else:
        print('dataloader stage problem')

    return dataloader
