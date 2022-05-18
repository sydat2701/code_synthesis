

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from keras import optimizers
#from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


from keras.models import Sequential, Model, Input
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
#import tensor as tf


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



# Create a model
# model = keras.Sequential([
#     keras.layers.Dense(128, activation='sigmoid', input_shape=(86,)),
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(64, activation='sigmoid'),
#     keras.layers.Dropout(0.3),
#     keras.layers.Dense(32, activation='sigmoid'),
#     #keras.layers.Dropout(0.5),

#     keras.layers.Dense(32, activation='softmax')
# ])
#model.summary()

# plot training log
def plot_history(history):
    history_dict = history.history
    history_dict.keys()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'ro', label='Validation loss')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation loss/acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

#model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

checkpoint= ModelCheckpoint('/home/linhdt/Desktop/sydat/h5/naver_2_7_TVT.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
LR= ReduceLROnPlateau(monitor='val_loss', fator=0.1, verbose=1, patience= 120)
callbacks_list=[checkpoint, LR]

# Load and norminalize dataset
X_train = load_X('/home/linhdt/Desktop/sydat/datasetKP/X_train.txt')
#print(X_train)
Y_train = load_Y('/home/linhdt/Desktop/sydat/datasetKP/Y_train.txt')
X_train_norm = norm_X(X_train, augment=False)
#Y_train = np.concatenate([Y_train, Y_train]) #-------------------------------

X_test = load_X('/home/linhdt/Desktop/sydat/datasetKP/X_val.txt')
Y_test = load_Y('/home/linhdt/Desktop/sydat/datasetKP/Y_val.txt')

X_test_norm = norm_X(X_test)






def generateBatchImg(batch_size, data_path):
    f=open(data_path, "r")
    batch_img=np.zeros((8, 224,224,3))
    batch_label = np.zeros((8, 32))
    cnt=0
    while True:

        for row in f:
            #ges=row.split('/')[5].split('.')[1].lower()
            
            img=cv2.imread(row[:len(row)-1])
            img=cv2.resize(img, (224,224,3))
            img=img/255.0
            batch_img[cnt%batch_size, :, :,:]=img
            #batch_label[cnt%batch_size, ordLabel(ges)]=1
            cnt +=1
            if cnt==batch_size:
                cnt =0
                yield batch_img#, batch_label
                batch_img=np.zeros((8, 224,224,3))
                #batch_label = np.zeros((8, 32))

def generateBatchKP(batch_size, X_train_norm):
    #f=open(data_path, "r")
    batch_kp=np.zeros((8, 86))
    batch_label = np.zeros((8, 32))
    cnt=0
    while True:

        for row in X_train_norm:
            idx=0
            for ele in row:
                batch_kp[cnt%batch_size, idx]=ele
                idx +=1
            cnt +=1
            if cnt == batch_size:
                cnt=0
                #batch_kp=tf.convert_to_tensor(batch_kp, dtype=tf.int64)
                yield batch_kp
                batch_kp =np.zeros((8, 86))



def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)


    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(32)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(16)(x)
    x = Activation("relu")(x)

    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model

def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(128, input_shape=(86,), activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))

    # return our model
    return model




# create the MLP and CNN models
mlp = create_mlp(86, regress=False)
cnn = create_cnn(224, 224, 3, regress=False)
# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.outputs[0], cnn.outputs[0]])
# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(32, activation="relu")(combinedInput)
x = Dense(32, activation="softmax")(x)
# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[mlp.inputs[0], cnn.inputs[0]], outputs=x)
model.summary()


opt = Adam(lr=0.01, decay=1e-3 / 200)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)

model.fit(x=[X_train_norm, generateBatchImg(batch_size, '/home/linhdt/Desktop/sydat/datasetKP/train_img.txt')], y=Y_train, validation_data=([generateBatchKP(batch_size, X_test_norm), generateBatchImg(batch_size, '/home/linhdt/Desktop/sydat/datasetKP/val_img.txt')], Y_test), epochs=500, batch_size=batch_size, \
    verbose=1, callbacks= callbacks_list)

# history = model.fit(X_train_norm, Y_train, validation_data=(X_test_norm, Y_test), epochs=500, batch_size=32, verbose=1, callbacks= callbacks_list)

# plot_history(history)

# save model 
# model.save('pose_classifier_26_6.h5')

# # Testing on single example
# LABELS = ["STANDING", "BENDING", "CROUCHING"]

# X_sample = load_X('dataset/X_sample.txt')
# X_sample_norm = norm_X(X_sample)
# y_out = model.predict(X_sample_norm[0].reshape(1, 36))

# print("Estimated pose:")
# for idx in range(len(LABELS)):
#     print(LABELS[idx] + ": \t" + str(y_out[0][idx]))
# plot(X_sample[0])

# X_test_pre = load_X('/home/linhdt/Desktop/sydat/datasetKP/X_test.txt')
# Y_test_pre = load_Y('/home/linhdt/Desktop/sydat/datasetKP/Y_test.txt')
# X_test_norm_pre = norm_X(X_test_pre)

# model.evaluate(X_test_norm_pre, Y_test_pre, verbose=1)

# from keras.models import load_model
# model=load_model('naver_26_6(1).h5')
# model.load_weights('naver_26_6(1).h5')
# pred= model.predict(X_test_norm_pre[56].reshape(1, 86))
# print(np.argmax(pred))
# print('true: ', Y_test_pre[56])