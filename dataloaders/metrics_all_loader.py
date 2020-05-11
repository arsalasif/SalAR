from __future__ import division

import os
import numpy as np
import cv2
from scipy.misc import imresize

from dataloaders.helpers import *
from torch.utils.data import Dataset


class MetricsAllLoader(Dataset):
    """Loader for inference metrics file"""

    def __init__(self,
                 inputRes=None,
                 originalRes=None,
                 db_root_dir=None,
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892)):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.inputRes = inputRes
        self.originalRes = originalRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval

        fixation = []
        names = []
        gt_img = []
        fname = 'val_seqs_all' # test seq file
        with open(os.path.join(db_root_dir, fname + '.txt')) as f:
            seqs = f.readlines()
            img_list = []
            for seq in seqs:
                images = np.sort(os.listdir(os.path.join(db_root_dir, 'testing', seq.strip(), 'images')))
                images_path = list(map(lambda x: os.path.join('testing', seq.strip(), 'images', x), images))
                img_list.extend(images_path)
                lab = np.sort(os.listdir(os.path.join(db_root_dir, 'testing', seq.strip(), 'maps')))
                lab_path = list(map(lambda x: os.path.join('testing', seq.strip(), 'maps', x), lab))
                gt_img.extend(lab_path)
                fix_path_base = os.path.join(db_root_dir, 'testing', seq.strip(), 'fixation')
                fix = np.sort([f for f in os.listdir(fix_path_base) if os.path.isfile(os.path.join(fix_path_base, f))])
                fix_path = list(map(lambda x: os.path.join('testing', seq.strip(), 'fixation', x), fix))
                fixation.extend(fix_path)
                for i in range(len(fix_path)):
                    names.append(seq)


        assert (len(gt_img) == len(img_list))

        self.img_list = img_list
        self.fixation = fixation
        self.names = names
        self.gt_img = gt_img

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, fixation, seq_name, other_map, gt_img = self.make_img_gt_pair(idx)

        sample = {'image': img, 'fixation': fixation, 'other_map': other_map, 'gt_img': gt_img, 'name': seq_name}


        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        seq_name = self.names[idx]

        fixation = cv2.imread(os.path.join(self.db_root_dir, self.fixation[idx]), 0)
        gt_img = cv2.imread(os.path.join(self.db_root_dir, self.gt_img[idx]), 0)

        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))

        if self.inputRes is not None:
            img = imresize(img, self.inputRes)
        if self.originalRes is not None:
            fixation = imresize(fixation, self.originalRes, interp='nearest')
            gt_img = imresize(gt_img, self.originalRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        fixation_res = fixation.shape
        other_map = np.zeros(fixation_res)
        for i in range(0, 100):
            rand_frame = random.randint(0, len(self.fixation)-1)
            rand_fixation = cv2.imread(os.path.join(self.db_root_dir, self.fixation[rand_frame]), 0)
            other_map += imresize(rand_fixation, fixation_res, interp='nearest')
            other_map = np.clip(other_map, 0, 255)

        return img, fixation, seq_name, other_map, gt_img
