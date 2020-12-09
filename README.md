# Automatic-Segmentation-of-Temporal-Bone-Structures-from-Clinical-Conventional-CT



## Introduction

This is an open source code for temporal bone CT structures segmentation.

See more details in paper “Automatic Segmentation of Temporal Bone Structures from Clinical Conventional CT Using a CNN Approach”.

### Major features

##### Newly Design

​	We design a 3D CNN with fewer parameters referred to as W-Net with only 2013060 parameters. 

##### Small size

​	The model file generated by PyTorch is only 7.69Mb, which is suitable to be placed directly in GitHub code repository.

##### High Accuracy

​	Our method achieved human-level accuracy in the segmentation of the cochlear labyrinth, ossicular chain and facial nerve.

## Getting Started

### Environment

Python 3.6.7 or 3.7.x

Pytorch 1.0.1+

### Dataset

The dataset is contained in this repository. It can also be downloaded from https://bhpan.buaa.edu.cn:443/link/97FFE810B358D9395EAEA09CD096F149.


### Folder structure

│  .gitignore

│  dataset.py

│  ......

│  wnet.pth

│  wnet.py

│

├─test

│  ├─ 1

│  ├─ 2

│  ......

└─train

│  ├─1

│  ├─2

│  ......

### Train

Run main.py and choose "1".K-fold cross validation is used in the training. During the training process, the program will output the dice value on each structure. The trained model file will be saved in the automatically created "model" folder

### Test

Run "main.py" and choose "2". The results of automatic segmentation will be generated into the same folder with the data. The "prediction.nii.gz" file represents the out put of the network, and the "prediction_new.nii.gz" represents the results optimized by [Maximum Region Growing Approach](https://github.com/Dnkii/clean-the-nii).

