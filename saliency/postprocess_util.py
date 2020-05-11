import cv2
import torch
import os
import numpy as np
from IPython import embed

RESIZE = (480, 720) # H, W


def postprocess_prediction(prediction, size=None, blur=99):
    """
    Postprocess saliency maps by resizing and applying gaussian blurringself.

    args:
        prediction: numpy array with saliency postprocess_prediction
        size: original (H,W) of the image
    returns:
        numpy array with saliency map normalized 0-255 (int8)
    """
    saliency_map = (prediction * 255).astype(np.uint8)

    if size is None:
        size = RESIZE

    # resize back to original size
    saliency_map = cv2.resize(saliency_map, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    saliency_map = cv2.GaussianBlur(saliency_map, (blur, blur), 0)
    # clip again
    saliency_map = np.clip(saliency_map, 0, 255)

    return saliency_map


def normalize_map(s_map):
	# normalize the salience map (as done in MIT code)
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return norm_s_map
