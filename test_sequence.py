# Package Includes
from __future__ import division

import os
import socket
import timeit
from datetime import datetime

# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
from dataloaders import sequence_loader as db
from dataloaders import custom_transforms as tr
import networks.salar as salar
from dataloaders.helpers import *
from mypath import Path
from saliency.saliency_metrics import AUC_Judd, CC, NSS, SIM
import cv2
from saliency.postprocess_util import postprocess_prediction
from saliency.postprocess_util import normalize_map
import numpy as np
import imageio


def superimpose(image, heatmap):
    hmap = heatmap/heatmap.max()
    hmap = (hmap*255).astype(np.uint8)
    hmap = cv2.applyColorMap(hmap, 4)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = 0.3*img + 0.7*hmap
    img = (img).astype(np.uint8)
    return img

seq_name = 'Diving-Side-001'
save_dir = Path.save_root_dir()

# Select which GPU, -1 if CPU
gpu_id = 1
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))

VIS_RES = 1 # Visualize the results?
SAVE_RES = 1 # Save the results?
WRITE_RES = 1 # Write the results?

modelName = 'ucf_salar'

ucf = True
hollywood = False
dhf1k = False

if ucf:
    db_root_dir = './dataloaders/ucf'
    inputRes = (180, 320, 3)
elif hollywood:
    db_root_dir = './dataloaders/hollywood'
    inputRes = (180, 320, 3)
elif dhf1k:
    db_root_dir = './dataloaders/dhf1k'
    inputRes = (180, 320, 3)

print('Using model: ' + str(modelName))

HOME_PATH = './runs'
if ucf:
    print("ucf")
    HOME_PATH = os.path.join(HOME_PATH, 'ucf')
if hollywood:
    print("hollywood")
    HOME_PATH = os.path.join(HOME_PATH, 'hollywood')
if dhf1k:
    print("dhf1k")
    HOME_PATH = os.path.join(HOME_PATH, 'dhf1k')

OUTPUT_PATH = os.path.join(HOME_PATH, 'visualize')

# create output file
if not os.path.exists(OUTPUT_PATH):
	os.makedirs(OUTPUT_PATH)

# Network definition
net = salar.SalAR(pretrained=0)

checkpoint = torch.load(os.path.join(save_dir, modelName + '.pth'), map_location=lambda storage, loc: storage)
net.load_state_dict(checkpoint)

net.to(device)

# Testing dataset and its iterator
db_test = db.SequenceLoader(inputRes=inputRes, originalRes=None, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name)
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)


num_img_ts = len(testloader)
loss_tr = []
aveGrad = 0


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3)

anim_img = []
anim_gt = []
anim_pred = []
anim_len = len(testloader)
net = net.eval()
with torch.no_grad():
    animation = []
    for ii, sample_batched in enumerate(testloader):

        img, gt, fixation, img_orig = sample_batched['image'], sample_batched['gt'], sample_batched['fixation'], sample_batched['img_orig']

        # Forward of the mini-batch
        inputs = img.to(device)

        outputs = net.forward(inputs)
        for jj in range(int(inputs.size()[0])):
            pred = np.transpose(torch.relu(outputs[-1]).cpu().data.numpy()[jj, :, :, :], (1, 2, 0))

            pred = np.squeeze(pred)

            img_orig = np.transpose(img_orig.numpy()[jj, :, :, :], (1, 2, 0))

            prediction = normalize_map(pred)
            prediction = postprocess_prediction(prediction, (gt.shape[2], gt.shape[3]))
            prediction = normalize_map(prediction)
            prediction *= 255
            anim_img.append(img_orig)
            anim_gt.append(gt.squeeze().data.cpu().numpy())
            anim_pred.append(im_normalize(prediction))

def update_animation(i):
    ax[0].cla()
    ax[1].cla()
    ax[2].cla()
    ax[0].set_title('Image')
    ax[1].set_title('Ground Truth')
    ax[2].set_title('Prediction')
    ax[0].imshow(cv2.cvtColor(anim_img[i], cv2.COLOR_BGR2RGB))
    ax[1].imshow(superimpose(anim_img[i], anim_gt[i]))
    ax[2].imshow(superimpose(anim_img[i], anim_pred[i]))
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')

from matplotlib import animation
save_path = os.path.join(OUTPUT_PATH, seq_name + '.mp4')
FFwriter = animation.FFMpegWriter(fps=30, codec="libx264")
anim = animation.FuncAnimation(fig, update_animation, frames=anim_len, interval=100, save_count=anim_len)
anim.save(save_path, writer=FFwriter)

