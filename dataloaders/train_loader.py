from __future__ import division

import os
import numpy as np
import cv2
from scipy.misc import imresize

from dataloaders.helpers import *
from torch.utils.data import Dataset


class TrainLoader(Dataset):
    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir=None,
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892)):

        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval

        if self.train:
            fname = 'train_seqs_all'
        else:
            fname = 'val_seqs_all'

        if self.train:
            with open(os.path.join(db_root_dir, fname + '.txt')) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                for seq in seqs:
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'training', seq.strip(), 'images')))
                    images_path = list(map(lambda x: os.path.join('training', seq.strip(), 'images', x), images))
                    img_list.extend(images_path)
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'training', seq.strip(), 'maps')))
                    lab_path = list(map(lambda x: os.path.join('training', seq.strip(), 'maps', x), lab))
                    labels.extend(lab_path)
        else:
            with open(os.path.join(db_root_dir, fname + '.txt')) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                for seq in seqs:
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'testing', seq.strip(), 'images')))
                    images_path = list(map(lambda x: os.path.join('testing', seq.strip(), 'images', x), images))
                    img_list.extend(images_path)
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'testing', seq.strip(), 'maps')))
                    lab_path = list(map(lambda x: os.path.join('testing', seq.strip(), 'maps', x), lab))
                    labels.extend(lab_path)

        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)
        sample = {'image': img, 'gt': gt}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)

        if self.inputRes is not None:
            img = imresize(img, self.inputRes)
            label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        gt = np.array(label, dtype=np.float32)
        gt = gt/255.0
        gt = gt/np.max([gt.max(), 1e-8])

        return img, gt

