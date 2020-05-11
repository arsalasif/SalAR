from __future__ import division

import math
import os
from copy import deepcopy
import scipy.io

import torch
import torch.nn as nn
import torch.nn.modules as modules

from layers.salar_layers import center_crop, interp_surgery
from mypath import Path
from layers.gaussian_priors import GaussianPrior

class SalAR(nn.Module):
    def __init__(self, pretrained=1):
        super(SalAR, self).__init__()
        lay_list = [[64, 0.3, 64],
                    ['M', 128, 0.4, 128],
                    ['M', 256, 0.4, 256, 0.4, 256],
                    ['M', 512, 0.4, 512, 0.4, 512],
                    ['M', 512, 0.4, 512, 0.4, 512]]
        in_channels = [3, 64, 128, 256, 512]

        stages = modules.ModuleList()
        side_prep = modules.ModuleList()
        score_dsn = modules.ModuleList()
        upscale = modules.ModuleList()
        upscale_ = modules.ModuleList()
        attention_blocks = modules.ModuleList()
        gaussian_priors = modules.ModuleList()
        fuse_all = modules.ModuleList()
        # Construct the network
        for i in range(0, len(lay_list)):
            # Make the layers of the stages
            stages.append(make_layers_salar(lay_list[i], in_channels[i]))
            # Attention, side_prep and score_dsn start from layer 2
            if i > 0:
                # Make the layers of the preparation step
                side_prep.append(nn.Conv2d(lay_list[i][-1], 16, kernel_size=3, padding=1))

                # Make the layers of the score_dsn step
                score_dsn.append(nn.Conv2d(16, 1, kernel_size=1, padding=0))
                upscale_.append(nn.ConvTranspose2d(1, 1, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False))
                upscale.append(nn.ConvTranspose2d(16, 16, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False))

        # Attention
        for i in range(len(side_prep)-1):
            gaussian_priors.append(GaussianPrior(input_channels=16, nb_gaussian=16))
            attention_blocks.append(AttentionBlock(gating_channels=16, in_channels=16, inter_channels=16))
            fuse_all.append(Fuse())
        self.upscale = upscale
        self.upscale_ = upscale_
        self.stages = stages
        self.side_prep = side_prep
        self.score_dsn = score_dsn
        self.attention_blocks = attention_blocks
        self.gaussian_priors = gaussian_priors
        self.gaussian_prior_last = GaussianPrior(input_channels=16, nb_gaussian=16)
        self.fuse = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.fuse_all = fuse_all
        self.fuse_last = Fuse()
        #print("Initializing weights..")
        self._initialize_weights(pretrained)

    def forward(self, x):
        crop_h, crop_w = int(x.size()[-2]), int(x.size()[-1])
        x = self.stages[0](x)

        side = []
        side_out = []
        for i in range(1, len(self.stages)):
            x = self.stages[i](x)
            side_temp = self.side_prep[i - 1](x)
            side.append(center_crop(self.upscale[i - 1](side_temp), crop_h, crop_w))
            side_out.append(center_crop(self.upscale_[i - 1](self.score_dsn[i - 1](side_temp)), crop_h, crop_w))
        # attention modules
        for i in range(len(side)-1):
            side[i] = self.fuse_all[i](torch.cat((side[i], self.gaussian_priors[i](self.attention_blocks[i](side[-1], side[i]))), dim=1))

        side[-1] = self.fuse_last(torch.cat((side[-1], self.gaussian_prior_last(side[-1])), dim=1))
        out = torch.cat(side[:], dim=1)
        out = self.fuse(out)
        side_out.append(out)
        return side_out

    def _initialize_weights(self, pretrained):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.zero_()
                m.weight.data = interp_surgery(m)

        if pretrained == 2:
            print("Loading weights from Caffe VGG")
            # Load weights from Caffe
            caffe_weights = scipy.io.loadmat(os.path.join(Path.models_dir(), 'vgg_caffe.mat'))
            # Core network
            caffe_ind = 0
            for ind, layer in enumerate(self.stages.parameters()):
                if ind % 2 == 0:
                    c_w = torch.from_numpy(caffe_weights['weights'][0][caffe_ind].transpose())
                    assert (layer.data.shape == c_w.shape)
                    layer.data = c_w
                else:
                    c_b = torch.from_numpy(caffe_weights['biases'][0][caffe_ind][:, 0])
                    assert (layer.data.shape == c_b.shape)
                    layer.data = c_b
                    caffe_ind += 1


def find_conv_layers(_vgg):
    inds = []
    for i in range(len(_vgg.features)):
        if isinstance(_vgg.features[i], nn.Conv2d):
            inds.append(i)
    return inds


def make_layers_salar(cfg, in_channels):
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        elif v == 0.3 or v == 0.4:
            layers.extend([nn.Dropout2d(v)])
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v
    return nn.Sequential(*layers)


class AttentionBlock(nn.Module):
    def __init__(self, gating_channels, in_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class Fuse(nn.Module):
    def __init__(self):
        super(Fuse, self).__init__()
        self.convolution = nn.Conv2d(32, 16, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.convolution(x)
        x = self.relu(x)
        return x