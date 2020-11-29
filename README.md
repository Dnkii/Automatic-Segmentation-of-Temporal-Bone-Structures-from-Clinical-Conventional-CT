# Automatic-Segmentation-of-Temporal-Bone-Structures-from-Clinical-Conventional-CT

### Environment

Python 3.6.7 or 3.7.x

Pytorch 1.0.1+

### Dataset

Download from https://bhpan.buaa.edu.cn:443/link/2BBB146DCBECEAD19C7872DACC643BB9
（Available until：2021-12-31 23:59）

### Folder structure

│  .gitignore
│  dataset.py
│  LICENSE
│  main.py
│  maxregiongrowth.py
│  myloss_wce.py
│  nii.py
│  README.md
│  wnet.pth
│  wnet.py
│
├─test
│  ├─000_1
│  ├─001_1

│  ......

└─train
│  │  LICENSE
│  │
│  ├─000_1
│  ├─001_1
│  ......

### Train

Run main.py and insert "1".

### Test

Run main.py and insert "2".