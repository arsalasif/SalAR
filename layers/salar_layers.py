from __future__ import division

import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
from torch.nn import functional as F


def logit(x):
    return np.log(x/(1-x+1e-08)+1e-08)


def sigmoid_np(x):
    return 1/(1+np.exp(-x))

def kl_divergence(y_pred_, y_true_):
    y_true = torch.ge(y_true_, 0.5).float()
    y_pred = torch.relu(y_pred_)

    shape_c_out = y_true.shape[-1]  # width
    shape_r_out = y_true.shape[-2]  # height
    ep = 1e-07

    max_y_pred = torch.repeat_interleave(torch.unsqueeze(
        torch.repeat_interleave(torch.unsqueeze(torch.max(torch.max(y_pred, dim=2)[0], dim=2)[0], dim=-1), shape_r_out,
                                dim=-1), dim=-1), shape_c_out, dim=-1)

    sum_y_true = torch.repeat_interleave(
        torch.unsqueeze(torch.repeat_interleave(torch.unsqueeze(torch.sum(torch.sum(y_true, dim=2), dim=2), dim=-1),
                                                shape_r_out, dim=-1), dim=-1), shape_c_out, dim=-1)

    sum_y_pred = torch.repeat_interleave(torch.unsqueeze(
        torch.repeat_interleave(torch.unsqueeze(torch.sum(torch.sum((y_pred / (max_y_pred + ep)), dim=2), dim=2), dim=-1),
                                shape_r_out, dim=-1), dim=-1), shape_c_out, dim=-1)
    return torch.sum(torch.sum((y_true / (sum_y_true + ep)) * torch.log(
        ((y_true / (sum_y_true + ep)) / ((((y_pred / (max_y_pred + ep)) / (sum_y_pred + ep)) + ep) + ep)) + ep), dim=-1), dim=-1)


def center_crop(x, height, width):
    crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
    crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)
    # fixed indexing for PyTorch 0.4
    return F.pad(x, [int(crop_w.ceil()[0]), int(crop_w.floor()[0]), int(crop_h.ceil()[0]), int(crop_h.floor()[0])])


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


# set parameters s.t. deconvolutional layers compute bilinear interpolation
# this is for deconvolution without groups
def interp_surgery(lay):
        m, k, h, w = lay.weight.data.size()
        if m != k:
            print('input + output channels need to be the same')
            raise ValueError
        if h != w:
            print('filters need to be square')
            raise ValueError
        filt = upsample_filt(h)

        for i in range(m):
            lay.weight[i, i, :, :].data.copy_(torch.from_numpy(filt))

        return lay.weight.data