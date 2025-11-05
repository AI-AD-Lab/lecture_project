# ROD - devlepment project
## Introduction
This is the official PyTorch implementation of ROD: RGB-Only Fast and Efficient Off-road Freespace Detection
and will be upgraded

## Requirements
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

we made dockerfile and devconatiner

Install the same environment as [SAM](https://github.com/facebookresearch/segment-anything).

## Datasets
The ORFD dataset we used can be found at [ORFD](https://github.com/chaytonmin/Off-Road-Freespace-Detection). Extract and organize as follows:
```
|-- datasets
 |  |-- ORFD
 |  |  |-- training
 |  |  |  |-- sequence   |-- calib
 |  |  |                 |-- sparse_depth
 |  |  |                 |-- dense_depth
 |  |  |                 |-- lidar_data
 |  |  |                 |-- image_data
 |  |  |                 |-- gt_image
 ......
 |  |  |-- validation
 ......
 |  |  |-- testing
 ......
```

## Usage
### Image_Demo
```
python demo.py -> not working
```

## Acknowledgement
Our code is inspired by [EfficientSAM](https://github.com/yformer/EfficientSAM), [ORFD](https://github.com/chaytonmin/Off-Road-Freespace-Detection)
