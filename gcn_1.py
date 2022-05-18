import os
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

data_path = "/home/linhdt/Desktop/sydat/datasetKP_1"

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



def make_adj(adj_path):
    A= torch.zeros(43, 43)
    f=open(adj_path)
    for row in f:
        a1=int(row.split(' ')[0])
        a2=int(row.split(' ')[1])
        A[a1,a2]=1
    f.close()
    I = torch.eye(43).unsqueeze(0) #ma tran don vi
    A=A+I
    D = (torch.sum(A, 1) + 1e-5) ** (-0.5)
    A_norm= D.view( 43, 1) * A * D.view(1, 43)
    return  A_norm

class ImageDataset(Dataset):
    """Tabular and Image dataset."""

    def __init__(self, X_path, Y_path, adj_path):
        
        self.A_norm= make_adj(adj_path)
        #self.X_norm_pre= load_X(X_path)
        #self.X_norm = norm_X(self.X_norm_pre)
        self.X_norm = load_X(X_path)
        # print("X pre:", self.X_norm_pre)
        # print("X norm:", self.X_norm)

        self.Y_set=load_Y(Y_path)
        
        #print("rd_arr:", self.rd_arr)



    def __len__(self):
        return len(self.Y_set)

    def __getitem__(self, idx):
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", idx)
        #x=np.zeros((43,2))
        x= self.X_norm[idx].reshape(1,43,-1)
        x=torch.from_numpy(x).float()
        #print("________x", x.shape)
        #print("__________A", self.A_norm.shape)
        A=torch.bmm(self.A_norm, x)
        

        

        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # print("NOR-------------------------------------------------------------------------------------------")
        # print(idx)
        # print("-------------------------------------------------------------------------------------------")

        # #image = image/255.0 #********************************
        # image = image[..., :3]
        # image = transforms.functional.to_tensor(image)
        #print('###########################', image)

        # x_kp= torch.from_numpy( self.X_norm[self.rd_arr[idx]]).float()
        #x_kp= torch.from_numpy( self.X_norm[idx]).float()

        y= np.zeros((1,32))
        #y[0, self.Y_set[self.rd_arr[idx]]]=1
        y[0, self.Y_set[idx]]=1
        #y=self.Y_set[idx]
        #y= torch.Tensor(self.Y_set[idx])
        # y=y.view(1,-1)
        
        #y= torch.from_numpy(y)
        #print("y shape^^^^^^^^^^^^^^^^^^^^^^:", y.shape)
        #print(".")

                
        #return image.cuda(), x_kp.cuda(), y.cuda()
        return A, y




class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.fc1=nn.Linear(2,128)
        self.fc2=nn.Linear(128, 64)
        self.fc3=nn.Linear(64,32)
        self.fc4=nn.Linear(32,32)
        self.relu= nn.ReLU()
        self.sig=nn.Sigmoid()
        self.drop2=nn.Dropout(0.2)
        self.drop3=nn.Dropout(0.3)
        

    def forward(self, matrix):
        
        matrix=matrix.view(matrix.shape[0],43,-1)
        #print("##############",matrix.shape)
        matrix= self.fc1(matrix)
        matrix= self.relu(matrix)
        matrix= self.fc2(matrix)
        matrix=self.relu(matrix)
        matrix=self.fc3(matrix)
        #matrix=self.relu(matrix)
        #matrix= self.fc4(matrix)
        #print("UKKKKKKKKKKKKKKKKKKKKKKKK:", img.shape)
        
        #img=self.ln10(img)
        #print("OKKKKKKKKKKKKKKKKKKKKKKKk:", img.shape)
        #print("kp:", kp.shape)

        #x = torch.cat((img, kp), dim=1)
        #print("^^^^^^^^^^^^^^^^^", x.shape)
        
        # x = self.ln9(x)
        # #print("da den day1")        
        # x = self.relu(x)
        # #print("da den day2")        
        # x = self.ln8(x)
        # #print("da den day3")        
        # x = self.soft(x)#************************
        # #print("da den day4")
        

        return matrix





image_data_train = ImageDataset(os.path.join(data_path, 'X_train.txt'), os.path.join(data_path, 'Y_train.txt'), os.path.join(data_path, 'adj.txt'))
image_data_val = ImageDataset(os.path.join(data_path, 'X_val.txt'), os.path.join(data_path, 'Y_val.txt'), os.path.join(data_path, 'adj.txt'))
image_data_test = ImageDataset(os.path.join(data_path, 'X_test2.txt'), os.path.join(data_path, 'Y_test2.txt'), os.path.join(data_path, 'adj.txt'))


dataloader_train = DataLoader(dataset=image_data_train, batch_size=32, shuffle= True, num_workers=12)
dataloader_val = DataLoader(dataset=image_data_val, batch_size=32, shuffle= False, num_workers=12)
dataloader_test = DataLoader(dataset=image_data_test, batch_size=32, shuffle= False, num_workers=12)


# num_epoch=100


# model = Classifier()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(num_epoch):
#     epoch_train_acc = 0
#     epoch_train_loss=0
#     num_itera_train=0

#     for i, (inputs, labels) in enumerate(dataloader_train):
#         num_itera_train += 1
#         y_pred = model(inputs)
#         loss = criterion(y_pred, labels)
#         epoch_train_loss += loss

#         loss.backward()

#         optimizer.step()

#         optimizer.zero_grad()


#         train_acc = ((labels.argmax(1, keepdim=True)).eq(y_pred.argmax(1, keepdim=True)).sum()).float()/labels.shape[0]
#         epoch_train_acc +=train_acc
#     train_accuracy = epoch_train_acc / (num_itera_train)
#     train_loss = epoch_train_loss / num_itera_train

#     epoch_train_loss =0
#     epoch_train_acc=0
#     num_itera_train=0
#     print("epoch %d", (epoch+1))
#     print("train_acc: %.4f", train_acc, "train_loss:%.4f", train_loss)

#     #---------------------------------------VALIDATION---------------------------
#     epoch_val_acc = 0
#     epoch_val_loss=0
#     num_itera_val = 0
#     for i, (inputs, labels) in enumerate(dataloader_val):
#         num_itera_val += 1
#         y_pred = model(inputs)
#         loss = criterion(y_pred, labels)
#         epoch_val_loss += loss
        
#         val_acc = ((labels.argmax(1, keepdim=True)).eq(y_pred.argmax(1, keepdim=True)).sum()).float()/labels.shape[0]
#         epoch_val_acc +=val_acc
#     val_accuracy = epoch_val_acc / (num_itera_val)
#     val_loss = epoch_val_loss / num_itera_val

#     epoch_val_loss =0
#     epoch_val_acc=0
#     num_itera_val=0
#     print("train_acc: %.4f", val_acc, "train_loss:%.4f", val_loss)


# print("------------------------------TEST--------------------------------")

# epoch_test_acc = 0
# epoch_test_loss=0
# num_itera_test = 0
# for i, (inputs, labels) in enumerate(dataloader_test):
#     num_itera_test += 1
#     y_pred = model(inputs)
#     loss = criterion(y_pred, labels)
#     epoch_test_loss += loss
        
#     test_acc = ((labels.argmax(1, keepdim=True)).eq(y_pred.argmax(1, keepdim=True)).sum()).float()/labels.shape[0]
#     epoch_test_acc +=test_acc
# test_accuracy = epoch_test_acc / (num_itera_test)
# test_loss = epoch_test_loss / num_itera_test

# epoch_test_loss =0
# epoch_test_acc=0
# num_itera_test=0
# print("test_acc: %.4f", val_acc, "test_loss:%.4f", val_loss)


def transformLabels(tensLabel):
    tmp=tensLabel.numpy()
    res=torch.zeros(tmp.shape[0])
    for idx, item in enumerate(tmp):
        res[idx]=item[0]
    return res


def main():
    num_epoch=16


    model = Classifier()
    model=model.cuda()
    summary(model, (1,43,2), -1)
    #criterion = torch.nn.NLLLoss()
    #criterion = torch.nn.L1Loss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    #best_model_weights = copy.deepcopy(model.state_dict())
    best_val_loss= 10000
    for epoch in range(num_epoch):
        epoch_train_acc = 0
        epoch_train_loss=0
        num_itera_train=0

        for i, (inputs, labels) in enumerate(dataloader_train):
            #labels.squeeze(1)
            #labels=transformLabels(labels)
            labels=labels.view(labels.shape[0],-1).cuda()
            inputs=inputs.cuda()
            #inputs=torch.transpose(inputs,1,3).float().cuda()
            #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>", inputs)
            #print("^^^^^^^^^^^^^^^^", inputs.shape)
            # print("label shape********************8:", labels.shape)
            # print("^^^^^^^^^^^^^^^^^^^:", labels)
            #labels=labels.numpy()
            #np.expand_dims(labels,1)
            #labels=torch.from_numpy(labels)
            #print("labels^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^:", labels.shape)
            num_itera_train += 1
            #print("here 0")
            y_pred = model(inputs)
            #print("here 1")
            #y_pred = y_pred.numpy()
            #np.expand_dims(y_pred,1)
            #y_pred=torch.from_numpy(y_pred)
            #y_pred = y_pred.squeeze(1)
            #y_pred = y_pred.view(1,-1)
            #print("pred-------------------------------------:", y_pred.shape)
            #print("############################:", y_pred)
            #print(".")
            #loss = criterion(y_pred, labels)
            #print("############################:", y_pred.shape)
            #print("^^^^^^^^^^^^^^^^^^^^^^^^", labels.shape)
            
            loss= criterion(y_pred, labels.long())
            #print("here 0")
            epoch_train_loss += loss

            loss.backward()

            #print("'here 1")
            optimizer.step()

            optimizer.zero_grad()

            train_acc = ((labels.argmax(1, keepdim=True)).eq(y_pred.argmax(1, keepdim=True)).sum()).float()/labels.shape[0]
            #tmp=(labels.view(32,-1))
            #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>", tmp.shape)
            #train_acc = (tmp.eq(y_pred.argmax(1, keepdim=True)).sum()).float()/tmp.shape[0]
            
            #print("here 1")
            #------------------------------------------------------------------
            #ps = torch.exp(y_pred).data
            #print("here 1.5")
            #equality = (labels.data == ps.max(1)[1])
            #print("here 1.6")
            #train_acc = equality.type_as(torch.FloatTensor()).mean()
            #-------------------------------------------------------------------
            #print("here 2")
            epoch_train_acc +=train_acc
            #print("ok")
        #print("here 2.5")
        train_accuracy = epoch_train_acc / (num_itera_train)
        train_loss = epoch_train_loss / num_itera_train
        

        epoch_train_loss =0
        epoch_train_acc=0
        num_itera_train=0
        print("epoch", (epoch+1))
        print("train_acc:", train_accuracy.item(), "train_loss:", train_loss.item())

        #---------------------------------------VALIDATION---------------------------
        epoch_val_acc = 0
        epoch_val_loss=0
        num_itera_val = 0
        for i, (inputs, labels) in enumerate(dataloader_val):
            #labels.squeeze(1)
            #labels = transformLabels(labels)
            labels=labels.view(labels.shape[0],-1).cuda()
            inputs=inputs.cuda()
            num_itera_val += 1
            y_pred = model(inputs)
            loss = criterion(y_pred, labels.long())
            #sched.step()
            epoch_val_loss += loss
        
            val_acc = ((labels.argmax(1, keepdim=True)).eq(y_pred.argmax(1, keepdim=True)).sum()).float()/labels.shape[0]
            #tmp= (labels.view(32,-1))
            #val_acc = (tmp.eq(y_pred.argmax(1, keepdim=True)).sum()).float()/tmp.shape[0]
            #------------------------------------------------------------------
            #ps = torch.exp(y_pred).data
            #print("here 1.5")
            #equality = (labels.data == ps.max(1)[1])
            #print("here 1.6")
            #val_acc = equality.type_as(torch.FloatTensor()).mean()
            #-------------------------------------------------------------------

            epoch_val_acc +=val_acc
        val_accuracy = epoch_val_acc / (num_itera_val)
        val_loss = epoch_val_loss / num_itera_val

        
        epoch_val_loss =0
        epoch_val_acc=0
        num_itera_val=0
        print("val_acc:", val_accuracy.item(), "val_loss:", val_loss.item())
        if val_loss< best_val_loss:
            best_val_loss = val_loss
            #best_model_wts = copy.deepcopy(model.state_dict())
            # torch.save(model.state_dict(), '/home/linhdt/Desktop/sydat/model_weights_22_7.pth')
            torch.save(model.state_dict(), '/home/linhdt/Desktop/sydat/h5/model_weights_gcn.pth')
            print("----------------------->saved model<--------------------")
        sched.step()


    print("------------------------------TEST--------------------------------")

    epoch_test_acc = 0
    epoch_test_loss=0
    num_itera_test = 0
    for i, (inputs, labels) in enumerate(dataloader_test):
        #labels.squeeze(1)
        #labels=transformLabels(labels)
        labels=labels.view(labels.shape[0],-1).cuda()
        inputs=inputs.cuda()
        num_itera_test += 1
        y_pred = model(inputs)
        loss = criterion(y_pred, labels.long())
        epoch_test_loss += loss
        
        test_acc = ((labels.argmax(1, keepdim=True)).eq(y_pred.argmax(1, keepdim=True)).sum()).float()/labels.shape[0]
        #tmp=(labels.view(32,-1))
        #test_acc = (tmp.eq(y_pred.argmax(1, keepdim=True)).sum()).float()/tmp.shape[0]
        
        #------------------------------------------------------------------
        #ps = torch.exp(y_pred).data
        #print("here 1.5")
        #equality = (labels.data == ps.max(1)[1])
        #print("here 1.6")
        #test_acc = equality.type_as(torch.FloatTensor()).mean()
        #-------------------------------------------------------------------
        
        epoch_test_acc +=test_acc
    test_accuracy = epoch_test_acc / (num_itera_test)
    test_loss = epoch_test_loss / num_itera_test

    epoch_test_loss =0
    epoch_test_acc=0
    num_itera_test=0
    print("test_acc:", test_accuracy.item(), "test_loss:", test_loss.item())

if __name__=='__main__':
    main()
