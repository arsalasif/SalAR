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
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Custom includes
from dataloaders import train_loader as db
from dataloaders import custom_transforms as tr
import networks.salar as salar
from layers.salar_layers import kl_divergence
from mypath import Path

def get_lr(gamma, optimizer):
    return [group['lr'] * gamma
            for group in optimizer.param_groups]


# Select which GPU, -1 if CPU
gpu_id = 1
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))

# Network definition
nEpochs = 100 # Number of epochs
resume_epoch = 0 # Default is 0, change if want to resume

# # Setting other parameters
ucf = True
hollywood = False
dhf1k = False

snapshot = 1  # Store a model every snapshot epochs
nAveGrad = 10
save_dir = Path.save_root_dir()
if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))

if ucf:
    db_root_dir='./dataloaders/ucf'
    inputRes = (180, 320, 3)
elif hollywood:
    db_root_dir = './dataloaders/hollywood'
    inputRes = (180, 320, 3)
elif dhf1k:
    db_root_dir = './dataloaders/dhf1k'
    inputRes = (180, 320, 3)

modelName = 'ucf_salar'
resumeModelName = 'ucf_salar'


print(inputRes)
print(modelName)


if resume_epoch == 0:
    net = salar.SalAR(pretrained=2)
else:
    print('Resume model: ' + resumeModelName)
    net = salar.SalAR(pretrained=0)


net.to(device)

lr = 1e-4
print(lr)
gamma = 0.1
reduceLR = False
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

if resume_epoch != 0:
    print("Updating weights from: {}".format(
        os.path.join(save_dir, resumeModelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    checkpoint = torch.load(os.path.join(save_dir, resumeModelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if reduceLR:
        for param_group, lr in zip(optimizer.param_groups, get_lr(gamma, optimizer)):
            param_group['lr'] = lr
            print("Reduced lr to " + str(lr))



# Preparation of the data loaders
# Define augmentation transformations as a composition
composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                          tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                          tr.ToTensor()])
# Training dataset and its iterator
db_train = db.TrainLoader(train=True, inputRes=inputRes, db_root_dir=db_root_dir, transform=composed_transforms)
trainloader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=2)

# Testing dataset and its iterator
db_test = db.TrainLoader(train=False, inputRes=inputRes, db_root_dir=db_root_dir, transform=tr.ToTensor())
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2)

num_img_tr = len(trainloader)
num_img_ts = len(testloader)
running_loss_tr = [0] * 5
running_loss_ts = [0] * 5
loss_tr = []
loss_ts = []
aveGrad = 0

print("Training Network")
# Main Training and Testing Loop
for epoch in range(resume_epoch, nEpochs):
    start_time = timeit.default_timer()
    # One training epoch
    net = net.train()
    for ii, sample_batched in enumerate(trainloader):

        inputs, gts = sample_batched['image'], sample_batched['gt']

        # Forward-Backward of the mini-batch
        inputs.requires_grad_()
        inputs, gts = inputs.to(device), gts.to(device)

        outputs = net.forward(inputs)

        # Compute the losses, side outputs and fuse
        losses = [0] * len(outputs)
        for i in range(0, len(outputs)):
            losses[i] = kl_divergence(outputs[i], gts)
            running_loss_tr[i] += losses[i].item()
        # loss = (1 - epoch / (2 * nEpochs))*sum(losses[:-1]) + losses[-1]
        loss = sum(losses)
        # Print stuff
        if ii % num_img_tr == num_img_tr - 1:
           running_loss_tr = [x / num_img_tr for x in running_loss_tr]
           loss_tr.append(running_loss_tr[-1])
           print('[Epoch: %d, numImages: %5d]' % (epoch, ii + 1))
           for l in range(0, len(running_loss_tr)):
               print('Loss %d: %f' % (l, running_loss_tr[l]))
               running_loss_tr[l] = 0

           stop_time = timeit.default_timer()
           print("Execution time: " + str((stop_time - start_time)/60.0))

        loss /= nAveGrad
        loss.backward()
        aveGrad += 1

        # Update the weights once in nAveGrad forward passes
        if aveGrad % nAveGrad == 0:
            optimizer.step()
            optimizer.zero_grad()
            aveGrad = 0

    # Save the model
    if epoch == 0 or (epoch % snapshot) == snapshot - 1:

        torch.save({'optimizer_state_dict': optimizer.state_dict(), 'model_state_dict': net.state_dict()}, os.path.join(save_dir, modelName + '_epoch-' + str(epoch) + '.pth'))

    net = net.eval()
#    One testing epoch
    with torch.no_grad():
        start_time = timeit.default_timer()
        for ii, sample_batched in enumerate(testloader):
            inputs, gts = sample_batched['image'], sample_batched['gt']

            # Forward pass of the mini-batch
            inputs, gts = inputs.to(device), gts.to(device)

            outputs = net.forward(inputs)
            # Compute the losses, side outputs and fuse
            losses = [0] * len(outputs)
            for i in range(0, len(outputs)):
                losses[i] = kl_divergence(outputs[i], gts)
                running_loss_ts[i] += losses[i].item()
            # loss = (1 - epoch / (2 * nEpochs)) * sum(losses[:-1]) + losses[-1]
            loss = sum(losses)

            # Print stuff
            if ii % num_img_ts == num_img_ts - 1:
                running_loss_ts = [x / num_img_ts for x in running_loss_ts]
                loss_ts.append(running_loss_ts[-1])

                print('[Epoch: %d, numImages: %5d]' % (epoch, ii + 1))
                for l in range(0, len(running_loss_ts)):
                    print('***Testing *** Loss %d: %f' % (l, running_loss_ts[l]))
                    running_loss_ts[l] = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str((stop_time - start_time)/60.0))
