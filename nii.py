import pydicom
import os
import numpy as np
import cv2
import SimpleITK as sitk

def nii2array(Pathnii):
    img = sitk.ReadImage(Pathnii)
    data = sitk.GetArrayFromImage(img)
    # print("Shape:",data.shape)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    return data,spacing,origin

def savenii(vol0,spacing,origin,outname,std=False):
    if not std:
        vol = np.transpose(vol0, (2, 0, 1))
        vol = vol[::-1]
    else:
        vol=vol0
    out = sitk.GetImageFromArray(vol)
    out.SetSpacing(spacing)
    out.SetOrigin(origin)
    sitk.WriteImage(out,'%s.nii.gz'%(outname))

