import numpy as np
import torch
import argparse
from myloss_wce import MyLoss
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.nn as nn
from torchvision.transforms import transforms
from dataset import Make_Dataset,Make_Dataset_test,make_dataset_train
import matplotlib.pyplot as plt
import scipy.misc
import time
from os.path import basename
import os
import pandas as pd
import cv2
from wnet import wnet
import gc
from sklearn.model_selection import KFold
from nii import savenii
from maxregiongrowth import RegionGrowthOptimize as rgo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
])


y_transforms = transforms.Compose([
    transforms.ToTensor(),
])

parse=argparse.ArgumentParser()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path+'Dir created')
    else:
        print(path+'Dir Existed')


def datestr():
    now = time.localtime()
    return '{}{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday)

def timestr():
    now = time.localtime()
    return '{:02}{:02}'.format(now.tm_hour, now.tm_min)

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 1e-6:
            param_group['lr'] *= decay_rate

def train_model(criterion, train_dataloaders,test_dataloaders, k,kfold=5,num_epochs=20):
    model_path = 'model/wnet/w1d01/%s_%s/' %(datestr(),k)
    mkdir(model_path)
    model = wnet(1,4, batch_norm=False, sample=False).to(device)
    optimizer = optim.Adam(model.parameters())
    model.apply(weights_init)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    train_loss_list = []
    train_dice_list = []
    epoch_loss_list = []
    test_loss_list = []
    test_dice_list = []
    total_step = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        dt_size = len(train_dataloaders.dataset)
        # dt_size_test = len(dataloaders_test.dataset)
        epoch_loss = 0
        epoch_dice=0
        step = 0
        for x,y in train_dataloaders:
            step += 1
            inputs = x.to(device).float()
            inputs.requires_grad_()
            labels = y.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss,dices = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
            # train_dice_list.append(dices)
            epoch_loss += loss.item()
            epoch_dice += dices
            print("%d/%d,train_loss:%0.6f" % (step, (dt_size - 1) // train_dataloaders.batch_size + 1, loss.item()))
        
        test_loss=0
        test_dice=0
        model.eval()
        step_test=0
        for x,y in test_dataloaders:#
            step_test += 1
            inputs = x.to(device).float()
            labels = y.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss,dices = criterion(outputs, labels)
            loss.backward()
            test_loss+=loss.item()
            test_dice += dices
            print("%d/%d,test_loss:%0.6f" % (step_test, (len(test_dataloaders.dataset) - 1) // test_dataloaders.batch_size + 1, loss.item()))
        epoch_dice/=(len(train_dataloaders.dataset)/train_dataloaders.batch_size)
        test_dice/=(len(test_dataloaders.dataset)/test_dataloaders.batch_size)
        model.train()
        epoch_loss_list.append(epoch_loss)
        train_dice_list.append(epoch_dice.tolist())
        test_loss_list.append(test_loss*(kfold-1))
        test_dice_list.append(test_dice.tolist())
        step_loss = pd.DataFrame({'step': range(len(train_loss_list)), 'step_loss': train_loss_list})
        step_loss.to_csv(model_path + '/' + 'step_loss.csv',index=False)
        adjust_learning_rate(optimizer)
        print("epoch %d loss:%0.3f,test_loss:%0.3f" % ((epoch+1), epoch_loss, test_loss*(kfold-1)))
        if epoch % 5 == 4:
            torch.save(model.state_dict(), (model_path + '/%s_epoch_%d.pth' %(timestr(),(epoch+1))))
    plt.plot(epoch_loss_list,label="train")
    plt.plot(test_loss_list,label="test")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(bbox_to_anchor=(0.8, 0.97), loc=2, borderaxespad=0.)
    plt.savefig(model_path+"/accuracy_loss%s.jpg"%k)
    plt.close()
    train_dice_list = np.array(train_dice_list)
    test_dice_list = np.array(test_dice_list)
    ifnotsave=1
    while ifnotsave:
        try:
            step_dice = pd.DataFrame({'step': range(len(epoch_loss_list)), 
            'train_loss': epoch_loss_list, 'test_loss': test_loss_list,
            'train_dice_1': train_dice_list[:,0], 'train_dice_2': train_dice_list[:,1],
            'train_dice_3': train_dice_list[:,2],'test_dice_1': test_dice_list[:,0],
            'test_dice_2': test_dice_list[:,1],'test_dice_3': test_dice_list[:,2]})
            step_dice.to_csv(model_path + '/' + 'epoch_dice.csv',index=False)
            ifnotsave=0
        except:
            input("Failed to write info into epoch_dice.csv")
    del optimizer
    del model
    return

#训练模型
def train(train_path="train",model_path = "",test_path=""):

    batch_size = 2
    criterion = MyLoss()
    datasets=make_dataset_train(train_path)
    kfold = 5
    kf = KFold(n_splits=kfold)
    k=0
    for trainsets, testsets in kf.split(datasets):
        # if k==0:
        trainsets = np.array(datasets)[trainsets].tolist()
        testsets = np.array(datasets)[testsets].tolist()
        train_data = Make_Dataset(trainsets,transform=x_transforms,mask_transform=y_transforms)
        train_dataloaders = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_data = Make_Dataset(testsets,transform=x_transforms,mask_transform=y_transforms)
        test_dataloaders = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

        train_model(criterion, train_dataloaders,test_dataloaders, k, kfold)
        k+=1

def segonly(test_path="test",model_path='wnet.pth',mod="wnet",k=""):
    if mod =="wnet":
        from wnet import wnet
        model = wnet(1,4,batch_norm=True,sample=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0)))
    datasets=make_dataset_train(test_path)
    test_data = Make_Dataset_test(datasets,transform=x_transforms,mask_transform=y_transforms)
    test_dataloaders = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    folderlist = os.listdir(test_path)
    model.eval()
    dt_size = len(test_dataloaders.dataset)
    save_path = test_path
    i=0
    threshold = 0.2
    lossall=0
    for image,spc,ori in test_dataloaders:
        y = model(image.to(device))     
        folder_path = save_path + '/' + folderlist[i]
        y = y.view((4,80,64,64))
        # image_np=(image_np.cpu().numpy()*255).astype(np.uint8)
        predict = torch.detach(y).cpu().numpy()
        # predict = np.transpose(predict,[2,1,0])
        out=np.zeros((80,64,64))
        out[predict[1]>threshold]=1
        out[predict[2]>threshold]=2
        out[predict[3]>threshold]=3
        out = np.transpose(out,[0,2,1])
        spc=(spc[0].item(),spc[1].item(),spc[2].item())
        ori=(ori[0].item(),ori[1].item(),ori[2].item())
        savenii(out,spc,ori,'%s/predict%s'%(folder_path,k),std=True)
        rgo('%s/predict%s.nii.gz'%(folder_path,k))
        # savenii(out,spc,ori,'%s/predict%s'%(folder_path),std=True)
        # rgo('%s/predict.nii.gz'%(folder_path))
        print('Finish:%s/predict'%folder_path)
        i+=1

if __name__ == '__main__':
    mod = input("1 for Train,2 for Test,please insert 1 or 2:")
    if mod == "1":
        train_path = "train"
        train(train_path)
    elif mod == "2":
        modelpath = "wnet.pth" 
        mod="wnet"
        niipath = "test"
        segonly(niipath,modelpath,mod,"")