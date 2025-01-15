import torch
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader
from tool.utils import load_hdr_as_tensor
import os
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw
from scipy.io import loadmat
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib

matplotlib.use('agg')
mat_version='v7'
def kernel_i():
    weights_data = [
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
    ]
    weights = np.asarray(weights_data)  # , np.int)
    return weights

# compute truth image
def compute_truth(speck,Training_path=''):
    N,C,H,W=speck.shape
    print(N,C,H,W)
    weight=[]
    h=kernel_i()
    for i in range(N):
        w=speck[0,0,:,:]/speck[i,0,:,:]
        w=np.log(w+1/w)-np.log(2)
        w=1/np.exp(w)
        w[np.where(w<0.9)]=0
        weight.append(w)
    weight=np.asarray(weight)
    print(np.max(weight),np.min(weight))
    print(weight.shape)
    aa=np.sum(weight,0)
    bb=np.sum(weight*speck[:,0,:,:],0)
    print(np.max(aa), np.min(aa),aa.shape)
    print(np.max(bb), np.min(bb), bb.shape)
    truth=bb/aa
    print(truth.shape,np.max(truth),np.min(truth))
    truth.tofile((Training_path+'/multi_temporal_truth.bin'))
    return truth

#####

def load_dataset(root_dir, redux=None, params=None, shuffled=False, single=False,drop_last=False):
    """Loads dataset and returns corresponding data loader."""

    dataset=AbstractDataset(root_dir,redux,params.crop_size,False)
    if single:
        return DataLoader(dataset,num_workers=20,pin_memory=True,batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset,num_workers=20,pin_memory=True, batch_size=params.batch_size, shuffle=shuffled,drop_last=drop_last)


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, redux=0, crop_size=128, clean_targets=False):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        # self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size
        self.clean_targets = clean_targets
        self.imgs = os.listdir(root_dir)


        if redux:
            self.imgs = self.imgs[:redux]

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h, band = img_list[0].shape
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))
            cropped_imgs.append(self.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs

    def crop(self, image, i, j, h, w):
        return image[i:i + h, j:j + w, :]

    def __getitem__(self, index):
        """Retrieves image from data folder."""

        # Load PIL image
        img_path = os.path.join(self.root_dir, self.imgs[index])

        if mat_version=='v7.3':
            img = h5py.File(img_path, 'r+')
            img = np.transpose(img[list(img.keys())[0]])
            img = np.array(img).astype(np.float32)
        else:
            img=loadmat(img_path)['data']
            img=img.astype(np.float32)

        # TODO:MODIFIED 20221117
        img[np.where(img>=13.4)]=13.4
        img[np.where(img<=-13.4)]=-13.4


        if len(img.shape) == 4:  # train or valid process
            img1 = img[0]
            img2 = img[1]
            img3=img[2]

            # Random square crop
            if self.crop_size != 0:
                img1 = self._random_crop([img1])[0]
                img2 = self._random_crop([img2])[0]
                img3 = self._random_crop([img3])[0]
            source = tvF.to_tensor(img1)
            target = tvF.to_tensor(img2)
            label=tvF.to_tensor(img3)
            return source, target, label

        else:  # test process
            if self.crop_size != 0:
                img = self._random_crop([img])[0]

            source = tvF.to_tensor(img)
            # target = tvF.to_tensor(img)
            return source


    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)
