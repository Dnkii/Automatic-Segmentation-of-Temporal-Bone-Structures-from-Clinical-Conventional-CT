# Automatic-Segmentation-of-Temporal-Bone-Structures-from-Clinical-Conventional-CT

### Environment

Python 3.6.7 or 3.7.x

Pytorch 1.0.1+

### Dataset

Download from https://bhpan.buaa.edu.cn:443/link/2BBB146DCBECEAD19C7872DACC643BB9

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

Run main.py and choose "1".K-fold cross validation is used in the training. During the training process, the program will output the dice value on each structure. The trained model file will be saved in the automatically created "model" folder

### Test

Run main.py and choose "2". The results of automatic segmentation will be generated to the same file as the data.

