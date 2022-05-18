

#import thu vien
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from PIL import Image
from PIL import ImageOps
from keras.preprocessing import image
from keras.preprocessing.image import load_img
import cv2
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input, DepthwiseConv2D, Conv2DTranspose, Add, GlobalAveragePooling2D, AveragePooling2D, SeparableConv2D
from keras.layers.experimental.preprocessing import Normalization
from keras import Model
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import random
import keras
import os
from keras.initializers import GlorotNormal
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model
from skimage.transform import resize as imresize


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)



np.random.seed(30)
import random as rn
rn.seed(30)
from keras import backend as K
import tensorflow as tf
tf.random.set_seed(30) # sửa lại thay vì ban đầu: tf.set_random_seed()

img_size = (256, 256)
num_classes = 2
batch_size = 4

train_path='D:\\Download\\dat_train\\DATA\\train'
val_path='D:\\Download\\dat_train\\DATA\\val'

imgTrain_path='/home/linhdt/Desktop/sydat/DATA/train/image'
maskTrain_path='/home/linhdt/Desktop/sydat/DATA/train/mask'

imgVal_path='/home/linhdt/Desktop/sydat/DATA/val/image'
maskVal_path='/home/linhdt/Desktop/sydat/DATA/val/mask'

flag=1

def generateBatch(img_fd, mask_fd):
    
    countBatch=0
    img_arr=os.listdir(img_fd)
    length=len(img_arr)

    if length%batch_size!=0:
        flag=0
    tmp1=length//batch_size
    tmp2=length%batch_size
        

    random.Random(23).shuffle(img_arr)
    #print(img_arr)
    count=0
    countBatch=0
    while True:
        x_train = np.zeros((batch_size,256,256,3))
        y_train=np.zeros((batch_size,256,256,1))
        x_aug = np.zeros((batch_size,256,256,3))
        y_aug= np.zeros((batch_size,256,256,1))
        for img in img_arr[:tmp1*batch_size]:
            img_path=os.path.join(img_fd,img)
            mask_path=os.path.join(mask_fd, img[:len(img)-3]+'png')

            image=cv2.imread(img_path)
            image.astype('float32')
            image=cv2.resize(image, img_size)
            image=image/255.*2-1
            x_train[count%batch_size,:,:,:]=image

            #--------------------------------------------------------
            if(countBatch%3==0):
            	image=cv2.imread(img_path).astype('float32')
            	image=cv2.resize(image, (256,256))
            	shifted = cv2.warpAffine(image, np.float32([[1, 0, np.random.randint(-30,30)],[0, 1, np.random.randint(-30,30)]]),(image.shape[1], image.shape[0]))
            	gray = cv2.cvtColor(shifted,cv2.COLOR_BGR2GRAY)
            	x0, y0 = np.argwhere(gray > 0).min(axis=0)
            	x1, y1 = np.argwhere(gray > 0).max(axis=0)
            	cropped=shifted[x0:x1,y0:y1,:]
            	image_resized=imresize(cropped,(256,256,3))
            	M = cv2.getRotationMatrix2D((256//2,256//2), np.random.randint(-10,10), 1.0)
            	rotated = cv2.warpAffine(image_resized, M, (256,256))
            	x_aug[count%batch_size,:,:,:]=rotated/255.0




        
            #--------------------------------------------------------------------------------------
            mask=cv2.imread(mask_path,0)
            mask.astype('float32')
            mask=np.where(mask>0,1,mask)
            mask=cv2.medianBlur(mask,5)
            mask=cv2.resize(mask, img_size)
            mask=mask.reshape(img_size+(1,))
            y_train[count%batch_size,:,:,:]=mask
            # if count<50:
            # print(img_path +"|"+mask_path)


            #------------------------------------------------------------------------------------------
            if(countBatch%3==0):
            	mask=cv2.imread(mask_path).astype('float32')
            	mask=cv2.resize(mask, (256,256))
            	shifted = cv2.warpAffine(mask, np.float32([[1, 0, np.random.randint(-30,30)],[0, 1, np.random.randint(-30,30)]]),(mask.shape[1], mask.shape[0]))
            	gray = cv2.cvtColor(shifted,cv2.COLOR_BGR2GRAY)
            	x0, y0 = np.argwhere(gray > 0).min(axis=0)
            	x1, y1 = np.argwhere(gray > 0).max(axis=0)
            	cropped=shifted[x0:x1,y0:y1,:]
            	image_resized=imresize(cropped,(256,256,1))
            	M = cv2.getRotationMatrix2D((256//2,256//2), np.random.randint(-10,10), 1.0)
            	rotated = cv2.warpAffine(image_resized, M, (256,256))
            	#rotated.reshape((256,256,1))
            	rotated=np.expand_dims(rotated,axis=2)	#--------------------------------------------
            	y_aug[count%batch_size,:,:,:]=rotated[:,:,:]/255.0


			#--------------------------------------------------------------------------------------------
            count += 1
            if count%batch_size==0 and count >0:
            	if countBatch%3==0:
            		x_train= np.concatenate([x_train, x_aug])
            		y_train = np.concatenate([y_train, y_aug])
            	countBatch += 1
            	yield x_train, y_train
            
        
        if flag==0 and count>=tmp1*batch_size:
            x_train = np.zeros((tmp2,256,256,3))
            y_train=np.zeros((tmp2,256,256,1))
            for img in img_arr[tmp1*batch_size:]:
                #print("^^^^^^^^^^",idx)
                img_path=os.path.join(img_fd,img)
                mask_path=os.path.join(mask_fd, img[:len(img)-3]+'png')

                image=cv2.imread(img_path)
                image.astype('float32')
                image=cv2.resize(image, img_size)
                image=image/255.*2-1
                x_train[count%batch_size,:,:,:]=image


                mask=cv2.imread(mask_path,0)
                mask.astype('float32')
                mask=np.where(mask>0,1,mask)
                mask=cv2.medianBlur(mask,5)
                mask=cv2.resize(mask, (256,256))
                mask=mask.reshape(img_size+(1,))
                y_train[count%batch_size,:,:,:]=mask

                count += 1
                

            count=0
            countBatch=0
            yield x_train, y_train
        






def my_IoU(y_true, y_pred):
    y_pred = K.argmax(y_pred)
    y_pred = K.cast(y_pred, 'float32')
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    intersection = K.sum(y_true * y_pred)
    IoU = (intersection+0.0000001) / (K.sum(y_true) + K.sum(y_pred) - intersection + 0.001)
    return IoU

from tensorflow.keras.applications import MobileNetV2
#Model

base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False)

conv_1x1 = Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer= 'he_normal')(base_model.get_layer('block_12_add').output)
conv_1x1 = BatchNormalization()(conv_1x1)
conv_rate6 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', depthwise_initializer= 'he_normal', pointwise_initializer='he_normal', dilation_rate=3)(base_model.get_layer('block_12_add').output)
conv_rate6 = BatchNormalization()(conv_rate6)
conv_rate12 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', depthwise_initializer= 'he_normal', pointwise_initializer='he_normal', dilation_rate=6)(base_model.get_layer('block_12_add').output)
conv_rate12 = BatchNormalization()(conv_rate12)
conv_rate18 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', depthwise_initializer= 'he_normal', pointwise_initializer='he_normal', dilation_rate=9)(base_model.get_layer('block_12_add').output)
conv_rate18 = BatchNormalization()(conv_rate18)
image_pooling = AveragePooling2D(pool_size=(16, 16))(base_model.get_layer('block_12_add').output)
image_pooling = Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer= 'he_normal')(image_pooling)
image_pooling = BatchNormalization()(image_pooling)
image_pooling = UpSampling2D(size=(16, 16), interpolation='bilinear')(image_pooling)

concat1 = Concatenate()([conv_1x1, conv_rate6, conv_rate12, conv_rate18, image_pooling])

conv5 = Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concat1)
conv5 = BatchNormalization()(conv5)
upsamp5 = UpSampling2D(size=(4, 4), interpolation='bilinear')(conv5)

conv6 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(base_model.get_layer('block_2_add').output)
conv6 = BatchNormalization()(conv6)

concat2 = Concatenate()([upsamp5, conv6])

conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', depthwise_initializer= 'he_normal', pointwise_initializer='he_normal')(concat2)
conv7 = BatchNormalization()(conv7)
conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', depthwise_initializer= 'he_normal', pointwise_initializer='he_normal')(conv7)
conv7 = BatchNormalization()(conv7)
conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', depthwise_initializer= 'he_normal', pointwise_initializer='he_normal')(conv7)
conv7 = BatchNormalization()(conv7)
upsamp_final = UpSampling2D(size=(4, 4), interpolation='bilinear')(conv7)
output = Conv2D(2, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(upsamp_final)

model = Model(inputs=base_model.input, outputs=output)
model.summary()

# model.load_weights('/content/drive/MyDrive/dlv3+_hand_weight_61+.h5')
# model.load_weights('/content/drive/MyDrive/Ket qua train model/cp5/handsegment_deeplab')
# def scheduler(epoch, lr):
#   if epoch==120:
#     lr=lr*0.1
#   if epoch==250:
#     lr=lr*0.1
#   return lr
# mylr = LearningRateScheduler(scheduler, verbose=1)
# model = load_model('/content/drive/MyDrive/temp.h5', custom_objects={'my_IoU': my_IoU})


model.compile(optimizer=Adam(learning_rate=0.00001), loss=SparseCategoricalCrossentropy(), metrics=[my_IoU])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10, min_lr=0.000001, verbose=1)

#image_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
#mask_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
seed=4
#train_gen1 = zip(image_datagen.flow(train_data_img, batch_size=batch_size, seed=seed), mask_datagen.flow(train_data_mask, batch_size=batch_size, seed=seed))
check=len(os.listdir(imgTrain_path))
if check%batch_size==0:
    steps = check // batch_size
else:
    steps = check//batch_size+1

check1=len(os.listdir(imgVal_path))
if check1%batch_size==0:
    steps_val=check//batch_size
else:
    steps_val=check//batch_size+1


#print('^^^^^^^^^^',steps)


my_callback = [tf.keras.callbacks.ModelCheckpoint(filepath='/home/linhdt/Desktop/sydat/humanSeg_30_5.h5', verbose=2, save_best_only=True, save_weights_only=True)]
model_history = model.fit(generateBatch(imgTrain_path, maskTrain_path), validation_data=generateBatch(imgVal_path, maskVal_path), steps_per_epoch=steps, validation_steps=steps_val ,batch_size=batch_size, epochs=100, verbose=1, callbacks=my_callback)

