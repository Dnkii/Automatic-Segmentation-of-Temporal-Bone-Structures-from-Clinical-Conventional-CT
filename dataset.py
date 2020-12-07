import torch.utils.data as data
import PIL.Image as Image
import os
from glob import glob
import cv2
import numpy as np
import torch
import SimpleITK as sitk
from nii import nii2array

def make_dataset_train(root):
    datasets = []
    for dirName,subdirList,fileList in os.walk(root):
        image_filelist = []
        mask_filelist = []
        for filename in fileList:
            if "input.nii.gz" in filename.lower(): 
                image_filelist.append(os.path.join(dirName,filename)) 
            if "mask.nii.gz" in filename.lower(): 
                mask_filelist.append(os.path.join(dirName,filename))

        if len(image_filelist)<1:
            continue
        datasets.append([image_filelist,mask_filelist])
    return datasets

class Make_Dataset(data.Dataset):
    def __init__(self, datasets, transform=None, mask_transform=None):
        self.datasets = datasets
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        x_path = self.datasets[index][0]
        y_path = self.datasets[index][1]
        image = sitk.ReadImage(x_path[0])
        image = sitk.GetArrayFromImage(image)
        mask = sitk.ReadImage(y_path[0])
        mask = sitk.GetArrayFromImage(mask)
        mask0=np.zeros((80,64,64))
        mask1=np.zeros((80,64,64))
        mask2=np.zeros((80,64,64))
        mask3=np.zeros((80,64,64))

        mask1[mask==1]=255
        mask2[mask==2]=255
        mask3[mask==3]=255
        mask0[(mask1+mask2+mask3)==0]=255

        image = np.transpose(image, [2,1,0])
        mask0 = np.transpose(mask0, [2,1,0])
        mask1 = np.transpose(mask1, [2,1,0])
        mask2 = np.transpose(mask2, [2,1,0])
        mask3 = np.transpose(mask3, [2,1,0])

        # if self.transform is not None:
        # image = image.astype(np.uint8)
        image = (image/image.max()).astype(np.float32)
        image = self.transform(image)
        height, width, depth = image.shape
        image = image.reshape(1, width, depth,height)
    # if self.mask_transform is not None:
        mask0 = mask0.astype(np.uint8)
        mask0 = self.mask_transform(mask0)
        mask0 = mask0.reshape(1, width, depth,height)
        mask1 = mask1.astype(np.uint8)
        mask1 = self.mask_transform(mask1)
        mask1 = mask1.reshape(1, width, depth,height)
        mask2 = mask2.astype(np.uint8)
        mask2 = self.mask_transform(mask2)
        mask2 = mask2.reshape(1, width, depth,height)
        mask3 = mask3.astype(np.uint8)
        mask3 = self.mask_transform(mask3)
        mask3 = mask3.reshape(1, width, depth,height)
        mask = torch.cat((mask0, mask1), dim=0)
        mask = torch.cat((mask, mask2), dim=0)
        mask = torch.cat((mask, mask3), dim=0)
        # print(mask.shape)
            # for i in range(64):
            #     sample = mask[:,:,i].numpy()
            #     cv2.imwrite('sample/%d.bmp'%i,sample)
            #     print('sample/%d.bmp'%i)
            # print(2)
        return image, mask

    def __len__(self):
        return len(self.datasets)

class Make_Dataset_test(data.Dataset):
    def __init__(self, datasets, transform=None, mask_transform=None):
        self.datasets = datasets
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        x_path = self.datasets[index][0]
        image,spc,ori = nii2array(x_path[0])
        image = np.transpose(image, [2,1,0])
        # if self.transform is not None:
        image = (image/image.max()*255).astype(np.uint8)
        image = self.transform(image)
        height, width, depth = image.shape
        image = image.reshape(1, width, depth,height)
    # if self.mask_transform is not None:

        return image,spc,ori

    def __len__(self):
        return len(self.datasets)
