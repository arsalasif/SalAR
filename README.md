# SalAR - Video Saliency

This is a trimmed version of the code. Full code with detailed documentation, and all models,  will be available soon.

### Dependencies:
Install - if necessary - the required dependencies:
 - Python (tested with 3.7.3, conda 4.6.14)
 - PyTorch (tested with PyTorch 1.1.0, CUDA 9.2, scipy 1.2.1)
 - Other python dependencies: numpy, scipy, matplotlib, opencv-python (cv2)
 - CUDA is required.


### Quick Inference
We have added a test sequence from the UCF-Sports dataset in the dataloaders folder.
1. Run `python test_sequence.py`.

### Inference
1. Download [Pre-trained UCF-Model](https://drive.google.com/open?id=1aCGeKugrPpXaF4bGCABNDt1VV17XczFI).
2. Download the model given for UCF-Sports dataset. Place it in `models` folder.
1. Download the datasets and place them under `dataloaders`. They can be downloaded from https://github.com/wenguanwang/DHF1K.
The dataset directory should look like the following:
```bash
└── dataloaders
│   ├── ucf
│   │   ├── training
│   │   	└── sequence_name
│   │   		├── images
│   │   		├── fixation
│   │   		└── maps
│   │   └── testing
│   │   	└── sequence_name
│   │   		├── images
│   │   		├── fixation
│   │   		└── maps
│   ├── hollywood
│   │   ├── training
│   │   	└── sequence_name
│   │   		├── images
│   │   		├── fixation
│   │   		└── maps
│   │   └── testing
│   │   	└── sequence_name
│   │   		├── images
│   │   		├── fixation
│   │   		└── maps
│   └── dhf1k
│   │   ├── training
│   │   	└── sequence_name
│   │   		├── images
│   │   		├── fixation
│   │   		└── maps
│   │   └── testing
│   │   	└── sequence_name
│   │   		├── images
│   │   		├── fixation
│   │   		└── maps
```
2. For DHF1K, please put the validation set files in the testing directory (the directory is named testing to streamline code).
3. Run `python inference_with_metrics.py`.
