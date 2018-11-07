# Pneumonia Detection

## Overview
Pytorch Implementation for pneumonia detection and localization using Faster R-CNN
The code is modified from [chenyuntc](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)'s simple-faster-rcnn-pytorch.
I replaced the RoIPooling module with RoIAlign and some other minor changes are implemented to train the pneumonia dataset.
Some prediction demo:
![image1]()

## Prerequisites
* Linux or OSX with NVIDIA GPU (Memory > 3.5G)
* CUDA 8.0, Pytorch 3.1, Python 3.6
* torchnet, visdom, cupy, cython
* skimage, matplotlib, sklearn, torchvision, tqdm

## Usage
1. Download Dataset
The dataset can be downloaded from
[Kaggle RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
There are around 26000 2D single channel CT images in the pneumonia dataset that provided in DICOM format.
The Faster R-CNN model is trained to predict the bounding box of the pneumonia area with a confidence score
2. Prepare Dataset
Convert DICOM file to PNG file and save in a specific folder(./stage_2_train/).
Read bounding box from 'stage_2_train_label.csv' and save each bounding box with the corresponding images
The folder should have the following structure.

```
# Root folder / image id / image.png or bbox.npy if any
/stage_2_train/0004cfab-14fd-4e49-80ba-63a80b6bddd6/image.png   # Training image
/stage_2_train/0004cfab-14fd-4e49-80ba-63a80b6bddd6/bbox.npy    # bbox of this image
/stage_2_train/00313ee0-9eaa-42f4-b0ab-c148ed3241cd/image.png   # Training image, does not have bbox
/stage_2_train/00322d4d-1c29-4943-afc9-b6754be640eb/image.png   # Training image, does not have bbox
...
```
3. Modify `root_dir` in `utils/Config.py`
4. Download Caffe pretrained model from [Google Drive](https://drive.google.com/drive/folders/1xz1cRK3em0kGNuUBKUplW_r2xtkO63S7?usp=sharing)
5. Specify the location of Caffe pretrained model `vgg16_caffe.pth` in `utils/Config.py`
5. Start visdom before training
```
nohup python -m visdom.server &
```

## Training Details
The 25000 CT images are split to the training set and testing set with ratio 9:1. There are 20197 out of 26000 images do not have
the corresponding bounding boxes because these subjects are healthy, which makes the failure of utilizing these images
for Faster R-CNN during training. Thus, these images are discarded during training.

## Evaluation
The training loss on the region proposal network and the Faster R-CNN core network is shown below. The results are evaluated on the mean average precision at the different intersection over union (IoU) thresholds.
Please refer to [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge#evaluation) for the details.
![image3]()

## Failure Cases
The dataset contains three categories of subjects, normal, pneumonia, and abnormal(cancer or other diseases) but only provides the bounding box for pneumonia images. However, the features of pneumonia and abnormal(cancer or other diseases)
are pretty similar, which caused the failure to distinguish pneumonia and abnormal images for Faster R-CNN. This results in
predicting bounding box for abnormal images.
![image4]()

## Acknowledgement
The code originates from [chenyuntc](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)'s simple-faster-rcnn-pytorch except some minor changes:
* Replaced the RoIPooling module with RoIAlign
* The convolution layers are modified to support binary classification
* Tried ResNet as the feature extraction network
* Tried histogram equalization during data preparation