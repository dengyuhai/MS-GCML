from __future__ import print_function
from __future__ import division

import os
import torch
import torchvision
import numpy as np
import PIL.Image

from .utils import make_transform

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform = None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []


    def nb_classes(self):
        debug_list=list(set(self.ys))
        # assert set(self.ys) == set(self.classes)
        return len(debug_list),debug_list

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        def img_load(index):
            # print(index,'\n')
            im = PIL.Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1 : im = im.convert('RGB') 
            if self.transform is not None:
                im = self.transform(im)
            return im

        im = img_load(index)
        target = self.ys[index]
        paths = self.im_paths[index]
        
        return im, paths,target, index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]
