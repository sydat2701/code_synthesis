from operator import mod
import os
from random import seed
import numpy as np
import cv2

import torch
from torch._C import set_flush_denormal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from torchsummary import summary
from numpy.lib.type_check import imag
import pandas as pd
import numpy as np
from PIL import Image
from skimage.transform import resize as imresize

from torch.utils.data import Dataset, DataLoader

import copy

data_path = "/home/linhdt/Desktop/sydat/datasetKP"

def ordLabel(x):
    if x=='thump up':
        return 0
    if x=='ok':
        return 1
    if x=='lucky flower':
        return 2
    if x=='chinese hello':
        return 3
    if x=="bird":
        return 4
    if x=='take photo':
        return 5
    if x=='hand up':
        return 6
    if x=='hand down':
        return 7
    if x=='i love you':
        return 8
    if x=='flower':
        return 9
    if x=='muscle':
        return 10
    if x=='cross hand over hand':
        return 11
    if x=='hi':
        return 12
    if x=='stop':
        return 13
    if x=='rock':
        return 14
    if x=='sleepy':
        return 15
    if x=='pray':
        return 16
    if x=='heart':
        return 17
    if x=='big heart':
        return 18
    if x=='small heart':
        return 19
    if x=='binocullar':
        return 20
    if x=='plus sign':
        return 21
    if x=='dab':
        return 22
    if x=='shut up':
        return 23
    if x=='rabbit':
        return 24
    if x=='pistol':
        return 25
    if x=='hand over head':
        return 26
    if x=='touch cheek':
        return 27
    if x=='touch head':
        return 28
    if x=='wait':
        return 29
    if x=='calling':
        return 30
    if x=='no_gesture' or x=='no gesture':
        return 31


batch_size=8

def load_X(X_path):
    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]], 
        dtype=np.float32
    )
    file.close()
    return X_

def load_Y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]], 
        dtype=np.int32
    )
    file.close()
    return y_

def euclidean_dist(a, b):
    # This function calculates the euclidean distance between 2 point in 2-D coordinates
    # if one of two points is (0,0), dist = 0
    # a, b: input array with dimension: m, 2
    # m: number of samples
    # 2: x and y coordinate
    try:
        if (a.shape[1] == 2 and a.shape == b.shape):
            # check if element of a and b is (0,0)
            bol_a = (a[:,0] != 0).astype(int)
            bol_b = (b[:,0] != 0).astype(int)
            dist = np.linalg.norm(a-b, axis=1)
            return((dist*bol_a*bol_b).reshape(a.shape[0],1))
    except:
        print("[Error]: Check dimension of input vector")
        return 0

def augment_kp(X):
    x_rd=[np.random.randint(-50,50)]*(X.shape[1]//2)
    y_rd=[np.random.randint(-50,50)]*(X.shape[1]//2)
    #print(x_rd)
    #print(y_rd)
    X_aug_x=X[:,0::2] + x_rd
    X_aug_y=X[:,1::2] + y_rd

    X_aug = np.column_stack((X_aug_x[:,:1], X_aug_y[:,:1]))
    for i in range(1, X.shape[1]//2):
        X_aug = np.column_stack((X_aug, X_aug_x[:,i:i+1], X_aug_y[:,i:i+1]))
    return X_aug

def norm_X(X, augment=False):
    num_sample = X.shape[0]
    # Keypoints

    Nose = X[:, 0*2:0*2+2]
    Lroot = X[:, 1*2:1*2+2]
    Rroot =X[:, 22*2:22*2+2]
    LCai1=X[:,2*2:2*2+2]
    LCai2=X[:, 3*2:3*2+2]
    LCai3=X[:, 4*2:4*2+2]
    LCai4=X[:, 5*2:5*2+2]
    
    LTro1=X[:, 6*2:6*2+2]
    LTro2=X[:, 7*2:7*2+2]
    LTro3=X[:, 8*2:8*2+2]
    LTro4=X[:, 9*2:9*2+2]
    
    LGiua1=X[:, 10*2:10*2+2]
    LGiua2=X[:, 11*2:11*2+2]
    LGiua3=X[:, 12*2:12*2+2]
    LGiua4=X[:, 13*2:13*2+2]

    LAU1=X[:, 14*2:14*2+2]
    LAU2=X[:, 15*2:15*2+2]
    LAU3=X[:, 16*2:16*2+2]
    LAU4=X[:, 17*2:17*2+2]
    
    LU1=X[:, 18*2:18*2+2]
    LU2=X[:, 19*2:19*2+2]
    LU3=X[:, 20*2:20*2+2]
    LU4=X[:, 21*2:21*2+2]
    
    #--------------------------------------------
    RCai1=X[:,23*2:23*2+2]
    RCai2=X[:, 24*2:24*2+2]
    RCai3=X[:, 25*2:25*2+2]
    RCai4=X[:, 26*2:26*2+2]
    
    RTro1=X[:, 27*2:27*2+2]
    RTro2=X[:, 28*2:28*2+2]
    RTro3=X[:, 29*2:29*2+2]
    RTro4=X[:, 30*2:30*2+2]
    
    RGiua1=X[:, 31*2:31*2+2]
    RGiua2=X[:, 32*2:32*2+2]
    RGiua3=X[:, 33*2:33*2+2]
    RGiua4=X[:, 34*2:34*2+2]
    
    RAU1=X[:, 35*2:35*2+2]
    RAU2=X[:, 36*2:36*2+2]
    RAU3=X[:, 37*2:37*2+2]
    RAU4=X[:, 38*2:38*2+2]
    
    RU1=X[:, 39*2:39*2+2]
    RU2=X[:, 40*2:40*2+2]
    RU3=X[:, 41*2:41*2+2]
    RU4=X[:, 42*2:42*2+2]
    
    
    #length of right hand
    length_root_RCai1= euclidean_dist(Rroot, RCai1)
    length_root_RCai2= euclidean_dist(Rroot, RCai2)
    length_root_RCai3= euclidean_dist(Rroot, RCai3)
    length_root_RCai4= euclidean_dist(Rroot, RCai4)
    
    length_root_RTro1= euclidean_dist(Rroot, RTro1)
    length_root_RTro2= euclidean_dist(Rroot, RTro2)
    length_root_RTro3= euclidean_dist(Rroot, RTro3)
    length_root_RTro4= euclidean_dist(Rroot, RTro4)
    
    length_root_RGiua1= euclidean_dist(Rroot, RGiua1)
    length_root_RGiua2= euclidean_dist(Rroot, RGiua2)
    length_root_RGiua3= euclidean_dist(Rroot, RGiua3)
    length_root_RGiua4= euclidean_dist(Rroot, RGiua4)
    
    length_root_RAU1= euclidean_dist(Rroot, RAU1)
    length_root_RAU2= euclidean_dist(Rroot, RAU2)
    length_root_RAU3= euclidean_dist(Rroot, RAU3)
    length_root_RAU4= euclidean_dist(Rroot, RAU4)
    
    length_root_RU1= euclidean_dist(Rroot, RU1)
    length_root_RU2= euclidean_dist(Rroot, RU2)
    length_root_RU3= euclidean_dist(Rroot, RU3)
    length_root_RU4= euclidean_dist(Rroot, RU4)
    
    
    #length of left hand
    length_root_LCai1= euclidean_dist(Rroot, LCai1)
    length_root_LCai2= euclidean_dist(Rroot, LCai2)
    length_root_LCai3= euclidean_dist(Rroot, LCai3)
    length_root_LCai4= euclidean_dist(Rroot, LCai4)
    
    length_root_LTro1= euclidean_dist(Rroot, LTro1)
    length_root_LTro2= euclidean_dist(Rroot, LTro2)
    length_root_LTro3= euclidean_dist(Rroot, LTro3)
    length_root_LTro4= euclidean_dist(Rroot, LTro4)
    
    length_root_LGiua1= euclidean_dist(Rroot, LGiua1)
    length_root_LGiua2= euclidean_dist(Rroot, LGiua2)
    length_root_LGiua3= euclidean_dist(Rroot, LGiua3)
    length_root_LGiua4= euclidean_dist(Rroot, LGiua4)
    
    length_root_LAU1= euclidean_dist(Rroot, LAU1)
    length_root_LAU2= euclidean_dist(Rroot, LAU2)
    length_root_LAU3= euclidean_dist(Rroot, LAU3)
    length_root_LAU4= euclidean_dist(Rroot, LAU4)
    
    length_root_LU1= euclidean_dist(Rroot, LU1)
    length_root_LU2= euclidean_dist(Rroot, LU2)
    length_root_LU3= euclidean_dist(Rroot, LU3)
    length_root_LU4= euclidean_dist(Rroot, LU4)
    
    
    
    #------------------------
    length_nose_Rroot= euclidean_dist(Nose, Rroot)
    length_nose_Lroot= euclidean_dist(Nose, Lroot)
    
    #------------------------
    length_left_hand= np.maximum.reduce([length_root_LCai1, length_root_LCai2, length_root_LCai3, length_root_LCai4, \
                                        length_root_LTro1, length_root_LTro2, length_root_LTro3, length_root_LTro4, length_root_LGiua1, \
                                        length_root_LGiua2, length_root_LGiua3, length_root_LGiua4, length_root_LAU1, length_root_LAU2, \
                                        length_root_LAU3, length_root_LAU4, length_root_LU1, length_root_LU2, length_root_LU3, length_root_LU4])
    
    length_right_hand= np.maximum.reduce([length_root_RCai1, length_root_RCai2, length_root_RCai3, length_root_RCai4, \
                                        length_root_RTro1, length_root_RTro2, length_root_RTro3, length_root_RTro4, length_root_RGiua1, \
                                        length_root_RGiua2, length_root_RGiua3, length_root_RGiua4, length_root_RAU1, length_root_RAU2, \
                                        length_root_RAU3, length_root_RAU4, length_root_RU1, length_root_RU2, length_root_RU3, length_root_RU4])

   
    length_final= np.maximum.reduce([length_left_hand, length_right_hand]) + np.maximum.reduce([length_nose_Rroot, length_nose_Lroot])
    length_chk=(length_final > 0 ).astype(int)
    keypoints_chk=(X>0).astype(int)
    
    chk=length_chk * keypoints_chk
    length_final[length_final==0]= 1
    
    #print(X[0])
    if augment:
        length_final_aug=length_final
        X_aug= augment_kp(X)
        print("X aug: **********: ", X_aug[0])
        keypoints_chk_aug=(X_aug>0).astype(int)
        chk_aug=length_chk * keypoints_chk_aug
        length_final_aug[length_final_aug==0]= 1
        
        
        num_pts_aug = (X_aug[:, 0::2] > 0).sum(1).reshape(num_sample,1)
        centr_x_aug = X_aug[:, 0::2].sum(1).reshape(num_sample,1) / num_pts_aug
        centr_y_aug = X_aug[:, 1::2].sum(1).reshape(num_sample,1) / num_pts_aug
        
        X_norm_x_aug = (X_aug[:, 0::2] - centr_x_aug) / length_final_aug
        X_norm_y_aug = (X_aug[:, 1::2] - centr_y_aug) / length_final_aug
        
        
        X_norm_aug = np.column_stack((X_norm_x_aug[:,:1], X_norm_y_aug[:,:1]))
        
        for i in range(1, X.shape[1]//2):
            X_norm_aug = np.column_stack((X_norm_aug, X_norm_x_aug[:,i:i+1], X_norm_y_aug[:,i:i+1]))
        
        
        X_norm_aug = X_norm_aug * chk_aug
        
        
    
    
    num_pts = (X[:, 0::2] > 0).sum(1).reshape(num_sample,1)
    centr_x = X[:, 0::2].sum(1).reshape(num_sample,1) / num_pts
    centr_y = X[:, 1::2].sum(1).reshape(num_sample,1) / num_pts
    
    
    X_norm_x = (X[:, 0::2] - centr_x) / length_final
    X_norm_y = (X[:, 1::2] - centr_y) / length_final
    
    

    
    X_norm = np.column_stack((X_norm_x[:,:1], X_norm_y[:,:1]))
        
    for i in range(1, X.shape[1]//2):
        X_norm = np.column_stack((X_norm, X_norm_x[:,i:i+1], X_norm_y[:,i:i+1]))
        
        
    X_norm = X_norm * chk
    
    if augment:
        X_norm = np.concatenate([X_norm, X_norm_aug])
    
    return X_norm



def handbox(img, height, width, coor, path, X):
    tg1=-100000000000
    tg2=100000000000
    xmin=tg2
    ymin=tg2
    xmax=tg1
    ymax=tg1

    xmin1=tg2
    ymin1=tg2
    xmax1=tg1
    ymax1=tg1
    try:
        img=cv2.resize(img,(height, width))
    except:
        print("!!!!!!!!!!!!!!!!!!!!!!!",path)
    cnt=0
    for _ in range(43):
        cnt +=1
        if cnt==1:
            continue
        if cnt>=2 and cnt <=22:
            x = coor[2*(cnt-1)]
            y = coor[2*(cnt-1) +1]

            if x>xmax:
                xmax=x
            if x<xmin:
                xmin=x
            if y>ymax:
                ymax=y
            if y<ymin:
                ymin=y
        elif cnt>=23 and cnt <= 43:
            x = coor[2*(cnt-1)]
            y = coor[2*(cnt-1) +1]
            if x>xmax1:
                xmax1=x
            if x<xmin1:
                xmin1=x
            if y>ymax1:
                ymax1=y
            if y<ymin1:
                ymin1=y
    esp=7
    ymin=ymin -esp
    xmin = xmin - esp
    ymin1= ymin1 -esp
    xmin1 -= esp
    xmax += esp
    xmax1 +=esp
    ymax +=esp
    ymax1 +=esp
    if xmin==xmax:
        xmin=xmin1
        xmax=xmax1
    if ymin==ymax:
        ymin=ymin1
        ymax=ymax1
    if xmin1==xmax1:
        xmin1=xmin
        xmax1=xmax
    if ymin1==ymax1:
        ymin1=ymin
        ymax1=ymax
    #print("_________>>>>>>>>", xmin, xmax, ymin, ymax, xmin1, xmax1, ymin1, ymax1)
    if ymax>width:
        ymax=width
    if ymax1>width:
        ymax1=width
    if xmax>height:
        xmax=height
    if xmax1>height:
        xmax1=height
    if xmin>height:
        #print(">>>>>>>>>>>>>>>>>>>")
        xmin=0
        xmax=2
    if xmin1>height:
        #print(">>>>>>>>>>>>>>>")
        xmin1=0
        xmax1=2
    if ymin>width:
        #print(">>>>>>>>>>>>>>>>>>>")
        ymin=0
        ymax=2
    if ymin1>width:
        #####print(">>>>>>>>>>>>>>>>>>>")
        ymin1=0
        ymax1=2
    if xmin<0:
        #print(">>>>>>>>>>>>>>>>>>>")
        xmin=0
    if xmin1<0:
        #print(">>>>>>>>>>>>>>>>>>>")
        xmin1=0
    if ymin<0:
        #print(">>>>>>>>>>>>>>>>>>>")
        ymin=0
    if ymin1<0:
        #print(">>>>>>>>>>>>>>>>>>>")
        ymin1=0

    
    imgCropLeft=img[int(ymin):int(ymax), int(xmin):int(xmax)]
    imgCropRight=img[int(ymin1):int(ymax1), int(xmin1):int(xmax1)]
    try:
        imgCropRight=cv2.resize(imgCropRight, (50,50))
        imgCropLeft=cv2.resize(imgCropLeft, (50,50))
    except:
        cv2.imshow('img', imgCropLeft)
        cv2.waitKey()
        cv2.destroyAllWindows()
    imgCrop = np.hstack((imgCropRight, imgCropLeft))
    #link= path.split('/')[-3]+'_'+ path.split('/')[-2]+'_'+path.split('/')[-1]
    #cv2.imwrite(os.path.join(hand_tar, link), imgCrop)
    # kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, 0]], np.float32) 

    # kernel = 1/3 * kernel

    # imgCrop = cv2.filter2D(imgCrop, -1, kernel)
    return imgCrop

class ImageDataset(Dataset):
    """Tabular and Image dataset."""

    def __init__(self, X_path, Y_path, img_path):
        
        self.X_norm_pre = load_X(X_path)
        self.X_norm = norm_X(self.X_norm_pre)
        # print("X pre:", self.X_norm_pre)
        # print("X norm:", self.X_norm)

        self.Y_set=load_Y(Y_path)

        f=open(img_path,"r")
        #self.img_set=np.array([t[:len(t)-1] for t in f])
        img_set=[]
        img_height=[]
        img_width=[]
        #cnt=0
        for t in f:
            #cnt +=1
            #print(cnt)
            #tmp=t.split(';')[0]
            img_set.append(t.split(';')[0])
            img_height.append(int(t.split(';')[1]))
            tmp=t.split(';')[2]
            img_width.append(int(tmp[:len(tmp)-1]))
        self.img_set=np.asarray(img_set)
        self.img_height=np.asarray(img_height)
        self.img_width=np.asarray(img_width)
        f.close()
        #print("rd_arr:", self.rd_arr)



    def __len__(self):
        return len(self.Y_set)

    def __getitem__(self, idx):

        try:
            image = cv2.imread(self.img_set[idx])
        except:
            image= cv2.imread(self.img_set[idx-1])
            idx = idx-1
            print(":::::::::::::::::::::",self.img_set[idx])

        # image = handbox(image, self.img_height[idx], self.img_width[idx], self.X_norm_pre[idx], self.img_set[idx], \
        #     self.X_norm_pre[idx])
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # print("NOR-------------------------------------------------------------------------------------------")
        # print(idx)
        # print("-------------------------------------------------------------------------------------------")

        # #image = image/255.0 #********************************
        # image = image[..., :3]
        image=cv2.resize(image, (50,100))
        image = transforms.functional.to_tensor(image)
        #print('###########################', image)

        # x_kp= torch.from_numpy( self.X_norm[self.rd_arr[idx]]).float()
        x_kp= torch.from_numpy( self.X_norm[idx]).float()

        #y= np.zeros((1,32))
        #y[0, self.Y_set[self.rd_arr[idx]]]=1
        #y[0, self.Y_set[idx]]=1
        y=self.Y_set[idx]
        #y= torch.Tensor(self.Y_set[idx])
        # y=y.view(1,-1)
        
        #y= torch.from_numpy(y)
        #print("y shape^^^^^^^^^^^^^^^^^^^^^^:", y.shape)
        #print(".")

                
        #return image.cuda(), x_kp.cuda(), y.cuda()
        return x_kp,image, y

def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)), nn.ReLU(), nn.BatchNorm2d(output_size), nn.MaxPool2d((2, 2)),
    )

    return block




from models.imagenet import mobilenetv2
class ChildClass(nn.Module):
    def __init__(self):
        super(ChildClass, self).__init__()
        #self.avg= nn.AdaptiveAvgPool2d(1)
        #self.maxP= nn.MaxPool2d(2)
        self.fc1= nn.Linear(1280, 256)
        self.relu=nn.ReLU()
        self.drop5= nn.Dropout2d(0.5)
        #self.drop6= nn.Dropout2d(0.6)
        #self.fc2= nn.Linear(128,32)
        self.fc3=nn.Linear(256,16)
        #self.drop2= nn.Dropout2d(0.2)
        
    def forward(self, x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.drop5(x)
        x=self.fc3(x)
        
        
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        model = mobilenetv2()
        model.load_state_dict(torch.load('pretrained/mobilenetv2-c5e733a8.pth'))
        model.classifier = ChildClass()
        self.model=model.cuda()
        
        self.conv1 = conv_block(3, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)
        #self.conv4 = conv_block(64, 128)
        
        self.flat=nn.Flatten()

        self.sig = nn.Sigmoid()
        # self.ln1 = nn.Linear(64*4*10, 16)
        self.ln1 = nn.Linear(64*26*26, 16)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout2d(0.5)
        self.dropout2=nn.Dropout2d(0.2)
        self.dropout3=nn.Dropout2d(0.3)
        self.dropout4=nn.Dropout2d(0.4)
        self.ln2 = nn.Linear(32, 16)

        self.ln3 = nn.Linear(86, 128)
        self.ln4 = nn.Linear(128, 64)
        self.ln5 = nn.Linear(64*6*14, 64)
        self.ln6 = nn.Linear(64, 32)
        self.ln7 = nn.Linear(32, 16)
        #self.ln7 = nn.Linear(32, 32)
        self.ln8 = nn.Linear(32,32)
        self.ln9 = nn.Linear(64,32)
        self.soft = nn.Softmax(dim=0)
        self.ln10= nn.Linear(1000,32)
        # model_eff = EfficientNet.from_pretrained('efficientnet-b0')
        # model_eff._fc=torch.nn.Linear(1280,32)
        # self.model_eff = model_eff.cuda
        

    def forward(self, kp, img):
        
        # img = self.conv1(img)
        # img= self.dropout2(img)
        # img = self.conv2(img)
        # img=self.dropout3(img)        
        # img = self.conv3(img)
        # img=self.dropout(img)     
        # img = img.reshape(img.shape[0], -1)
        # img = self.ln1(img)
        img=self.model(img)
        
        #print("^^^^^^^^^^^^^^^^^", img.shape)
        kp=self.ln3(kp)
        kp=self.sig(kp)
        kp=self.dropout2(kp)
        kp=self.ln4(kp)
        kp=self.sig(kp)
        kp=self.dropout3(kp)
        kp=self.ln6(kp)
        kp=self.sig(kp)
        kp=self.ln7(kp)
        
        #img=self.ln10(img)
        #print("OKKKKKKKKKKKKKKKKKKKKKKKk:", img.shape)
        #print("kp:", kp.shape)
        out= torch.cat([kp, img], dim=1)
        #out=kp+img
        out=self.ln8(out)

        

        return out











image_data_test = ImageDataset(os.path.join(data_path, 'X_test2.txt'), os.path.join(data_path, 'Y_test2.txt'), os.path.join(data_path, 'test2_img.txt'))

dataloader_test = DataLoader(dataset=image_data_test, batch_size=32, shuffle= False, num_workers=12)



def transformLabels(tensLabel):
    tmp=tensLabel.numpy()
    res=torch.zeros(tmp.shape[0])
    for idx, item in enumerate(tmp):
        res[idx]=item[0]
    return res

true=[]
predict=[]
def main():


    model = Classifier()
    #model.load_state_dict(torch.load('/home/linhdt/Desktop/sydat/h5/model_weights_thu_31_7_img_kp.pth', map_location="cpu"))
    checkpoint= torch.load('/home/linhdt/Desktop/sydat/h5/model_weights_thu_4_8_img_kp_min_loss.pth', map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cpu')
    model.eval()
    # model=model.cuda()
    # summary(model, (1,43,2), -1)
    # #criterion = torch.nn.NLLLoss()
    # #criterion = torch.nn.L1Loss()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # sched = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)
    # #best_model_weights = copy.deepcopy(model.state_dict())
    # best_val_loss= 10000

    print("------------------------------TEST--------------------------------")

    epoch_test_acc = 0
    epoch_test_loss=0
    num_itera_test = 0
    for i, (kp, img, labels) in enumerate(dataloader_test):
        #labels.squeeze(1)
        labels=transformLabels(labels).to('cpu')
        kp=kp.to('cpu')
        img.to('cpu')
        #labels=labels.view(labels.shape[0],-1)
        #inputs=inputs
        num_itera_test += 1
        y_pred = model(kp, img)
        
        
        #loss = criterion(y_pred, labels.long())
        #epoch_test_loss += loss
        
        #test_acc = ((labels.argmax(1, keepdim=True)).eq(y_pred.argmax(1, keepdim=True)).sum()).float()/labels.shape[0]
        #tmp=(labels.view(32,-1))
        #test_acc = (tmp.eq(y_pred.argmax(1, keepdim=True)).sum()).float()/tmp.shape[0]
        #true.append(labels.argmax(1))
        #------------------------------------------------------------------
        ps = torch.exp(y_pred).data
        #print("here 1.5")
        predict.append(ps.max(1)[1])
        #print(ps.argmax(1)[1])
        equality = (labels.data == ps.max(1)[1])
        #print("**************", y_pred[equality])
        #print("**********", equality)
        test_acc = equality.type_as(torch.FloatTensor()).mean()
        #-------------------------------------------------------------------
        
        epoch_test_acc +=test_acc
    test_accuracy = epoch_test_acc / (num_itera_test)
    #test_loss = epoch_test_loss / num_itera_test

    #epoch_test_loss =0
    epoch_test_acc=0
    num_itera_test=0
    print("==========================>test_acc:", test_accuracy.item())#, "test_loss:", test_loss.item())

if __name__=='__main__':
    main()
    #print(true)
    print("--------------------------------------------------")
    print(predict)