# Package Includes
from __future__ import division

import os
# PyTorch includes
import torch
from torch.utils.data import DataLoader
import timeit

# Custom includes
from dataloaders import metrics_all_loader as db
from dataloaders import custom_transforms as tr
import networks.salar as salar
from dataloaders.helpers import *
from mypath import Path
from saliency.saliency_metrics import AUC_Judd, AUC_shuffled, CC, NSS, SIM
import cv2
from saliency.postprocess_util import postprocess_prediction
from saliency.postprocess_util import normalize_map
import numpy as np

save_dir = Path.save_root_dir()

if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))

# Select which GPU, -1 if CPU
gpu_id = 1
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))


ucf = True
hollywood = False
dhf1k = False

SAVE_RES = True # Saves inferred results
WRITE_RES = True

if ucf:
    db_root_dir = './dataloaders/ucf'
    inputRes = (180, 320, 3)
elif hollywood:
    db_root_dir = './dataloaders/hollywood'
    inputRes = (180, 320, 3)
elif dhf1k:
    db_root_dir = './dataloaders/dhf1k'
    inputRes = (180, 320, 3)

modelName = 'ucf_salar'

print(inputRes)
print('Using model: ' + str(modelName))
HOME_PATH = './runs'
if ucf:
    HOME_PATH = os.path.join(HOME_PATH, 'ucf')
if hollywood:
    HOME_PATH = os.path.join(HOME_PATH, 'hollywood')
if dhf1k:
    HOME_PATH = os.path.join(HOME_PATH, 'dhf1k')

OUTPUT_PATH = os.path.join(HOME_PATH, 'results')

def get_metrics(gt, prediction, fixation, other_map, name, idx):
    gt = gt.squeeze().data.cpu().numpy()
    fixation = fixation.squeeze().data.cpu().numpy()
    other_map = other_map.squeeze().data.cpu().numpy()

    prediction = normalize_map(prediction)
    prediction = postprocess_prediction(prediction, size=(fixation.shape[0], fixation.shape[1]))
    prediction = normalize_map(prediction)
    prediction *= 255

    assert(prediction.shape == fixation.shape)

    mground_truth = gt.astype(np.float32)
    fground_truth = fixation.astype(np.float32)
    saliency_map = prediction.astype(np.float32)
    other_map = other_map.astype(np.float32)

    if SAVE_RES:
        seq_path = os.path.join(OUTPUT_PATH, name)
        if not os.path.exists(seq_path):
            os.makedirs(seq_path)
        if dhf1k:
            prediction_path = os.path.join(seq_path, str(idx).zfill(4) + '.png')
        if ucf or hollywood:
            prediction_path = os.path.join(seq_path, name + '_' + str(idx).zfill(3) + '.png')

        imageio.imsave(prediction_path, prediction)

    # Calculate metrics
    AUC_judd = AUC_Judd(saliency_map, fground_truth)
    sAUC = AUC_shuffled(saliency_map, fground_truth, other_map)
    nss = NSS(saliency_map, fground_truth)
    cc = CC(saliency_map, mground_truth)
    sim = SIM(saliency_map, mground_truth)
    return AUC_judd, sAUC, nss, cc, sim


# Network definition

net = newnet.SalAR(pretrained=0)
# net = salar.NewNetDropout(pretrained=0)

checkpoint = torch.load(os.path.join(save_dir, modelName + '.pth'),
                        map_location=lambda storage, loc: storage)
net.load_state_dict(checkpoint['model_state_dict'])

net.to(device)  # PyTorch 0.4.0 style

# Preparation of the data loaders
# Metrics dataset and its iterator
db_test = db.MetricsAllLoader(inputRes=inputRes, originalRes=None, db_root_dir=db_root_dir, transform=tr.ToTensor())
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2)

n = len(testloader)
print("N = " + str(n))
if WRITE_RES:
    path = os.path.join(HOME_PATH, 'metrics', str(modelName))
    if not os.path.exists(path):
        os.makedirs(path)
    metrics_result = open(path + "_results.txt", "w+")

final_seq_metric_list = []
final_metric_list = []
idx = 0
prev_name = ''
start_time = timeit.default_timer()
net = net.eval()
with torch.no_grad():  # PyTorch 0.4.0 style
    # Main Metrics Loop
    for ii, sample_batched in enumerate(testloader):

        img, name, fixation, other_map, gt_img = sample_batched['image'], sample_batched['name'], sample_batched['fixation'], sample_batched['other_map'], sample_batched['gt_img']

        # Forward of the mini-batch
        inputs = img.to(device)

        if prev_name == '':
            prev_name = name[0].rstrip()

        outputs = net.forward(inputs)
        if prev_name == name[0].rstrip() and ii != n-1:
            idx = idx + 1
        else:
            aucj = np.mean([y[0] for y in final_seq_metric_list])
            aucs = np.mean([y[1] for y in final_seq_metric_list])
            nss = np.mean([y[2] for y in final_seq_metric_list])
            cc = np.mean([y[3] for y in final_seq_metric_list])
            sim = np.mean([y[4] for y in final_seq_metric_list])

            if WRITE_RES:
                metrics_result.write("Final average of metrics for sequence " + prev_name + " is :\n")
                metrics_result.write("AUC-JUDD is {}\n".format(aucj))
                metrics_result.write("AUC-Shuffled is {}\n".format(aucs))
                metrics_result.write("NSS is {}\n".format(nss))
                metrics_result.write("CC is {}\n".format(cc))
                metrics_result.write("SIM is {}\n".format(sim))
            final_seq_metric_list = []
            idx = 1
            prev_name = name[0].rstrip()

        pred = np.transpose(torch.relu(outputs[-1]).cpu().data.numpy()[0, :, :, :], (1, 2, 0))

        pred = np.squeeze(pred)
        aucj, aucs, nss, cc, sim = get_metrics(gt_img, pred, fixation, other_map, prev_name, idx)

        final_seq_metric_list.append((aucj,
                                      aucs,
                                      nss,
                                      cc,
                                      sim))
        # # if frame averages needed
        final_metric_list.append((aucj,
                                  aucs,
                                  nss,
                                  cc,
                                  sim))

Aucj = np.mean([y[0] for y in final_metric_list])
Aucs = np.mean([y[1] for y in final_metric_list])
Nss = np.mean([y[2] for y in final_metric_list])
Cc = np.mean([y[3] for y in final_metric_list])
Sim = np.mean([y[4] for y in final_metric_list])

print("Final average of metrics is:")
print("AUC-JUDD is {}".format(Aucj))
print("AUC-Shuffled is {}".format(Aucs))
print("NSS is {}".format(Nss))
print("CC is {}".format(Cc))
print("SIM is {}".format(Sim))
print('')
if WRITE_RES:
    metrics_result.write("Final average of metrics is:\n")
    metrics_result.write("AUC-JUDD is {}\n".format(Aucj))
    metrics_result.write("AUC-Shuffled is {}\n".format(Aucs))
    metrics_result.write("NSS is {}\n".format(Nss))
    metrics_result.write("CC is {}\n".format(Cc))
    metrics_result.write("SIM is {}\n\n".format(Sim))

if WRITE_RES:
    metrics_result.close()

stop_time = timeit.default_timer()
print("Execution time: " + str((stop_time - start_time)/60.0))
