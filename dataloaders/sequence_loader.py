from __future__ import division

import os
import numpy as np
import cv2
from scipy.misc import imresize

from dataloaders.helpers import *
from torch.utils.data import Dataset


class SequenceLoader(Dataset):
    """Loader for test sequence file"""

    def __init__(self,
                 inputRes=None,
                 originalRes=None,
                 db_root_dir=None,
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None):

        self.inputRes = inputRes
        self.originalRes = originalRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name

        names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'testing', str(seq_name), 'images')))
        img_list = list(map(lambda x: os.path.join('testing', str(seq_name), 'images', x), names_img))

        name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'testing', str(seq_name), 'maps')))
        labels = list(map(lambda x: os.path.join('testing', str(seq_name), 'maps', x), name_label))

        fix_path_base = os.path.join(db_root_dir, 'testing', str(seq_name), 'fixation')
        fix = np.sort([f for f in os.listdir(fix_path_base) if os.path.isfile(os.path.join(fix_path_base, f))])
        fixation = list(map(lambda x: os.path.join('testing', str(seq_name), 'fixation', x), fix))


        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels
        self.fixation = fixation


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt, fixation, img_orig = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt, 'fixation': fixation, 'img_orig': img_orig}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        img_orig = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        fixation = cv2.imread(os.path.join(self.db_root_dir, self.fixation[idx]), 0)
        gt = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)

        if self.inputRes is not None:
            img = imresize(img, self.inputRes)
        if self.originalRes is not None:
            fixation = imresize(fixation, self.originalRes, interp='nearest')
            label = imresize(gt, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        return img, gt, fixation, img_orig

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])

