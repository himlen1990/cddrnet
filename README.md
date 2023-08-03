# Exploring Compact and Efficient Neural Networks for Real-Time Semantic Segmentation

## Introduction
This is the official repository for CDDRNet. The code is modified based on [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) and [DDRNet.pytorch](https://github.com/chenjun2hao/DDRNet.pytorch).

## Reference Environment

- 2080Ti*2
- pytorch 1.8.1 
- see requirements

## Usage

### 0. Dataset
- (Cityscapes) Download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset, including gtFine (gtFine_trainvaltest.zip (241MB) + leftImg8bit_trainvaltest.zip (11GB)) and extra(leftImg8bit_trainextra.zip (44GB) + label generated from HRNet-Semantic-Segmentation [Download Link](https://drive.google.com/file/d/1zFH-COPzcFf-_khv_gUgELBiC-TfJwwH/view?usp=drive_link))
- (CamVid) Download the [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) dataset. including original images/labels and extra images (gtFine of Cityscapes) + labels [Download Link](https://drive.google.com/file/d/1kXzIxxmlKT2qj9fA-jEVzFskbdtbA9RH/view?usp=drive_link).
- You can also train the network without extra data. In this case, you will get a lower MIoU. (also train with the no_extra_train.lst in data/list/camvid and data/list/cityscapes)
- If you got an error while training, check if the paths contained in lists of `data/list` are correct for dataset images.

### 1. Train
````bash
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg experiments/camvid/cddrnet.yaml
````

### 2. Eval
TODO


### Problems
(Train in a Server)Import cv2 error: libGL.so.1: cannot open shared object file: No such file or directory
- pip install opencv-python-headless (should have the same version with opencv-python)