
import numpy as np
import os
#from scipy.misc import imread, imresize
import datetime
import os
import warnings
warnings.filterwarnings("ignore")
import abc
from sys import getsizeof


from scipy import misc
#from scipy.misc import imread
from matplotlib.pyplot import imread
#from PIL import Image.resize as imresize
from skimage.transform import resize as imresize

np.random.seed(30)
import random as rn
rn.seed(30)
from keras import backend as K
import tensorflow as tf
tf.random.set_seed(30)

# Commented out IPython magic to ensure Python compatibility.
import cv2
import matplotlib.pyplot as plt
# % matplotlib inline

from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.layers import Dropout

# zip_path = 'drive/MyDrive/DATA/data_final_detect.zip'
# !cp "{zip_path}" .
# !unzip -q data_final_detect.zip
# !rm data_final_detect.zip

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)



project_folder='/home/linhdt/Desktop/sydat/Data_Seg_H_12_6'

def plot(history):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,4))
    axes[0].plot(history.history['loss'])   
    axes[0].plot(history.history['val_loss'])
    axes[0].legend(['loss','val_loss'])

    axes[1].plot(history.history['categorical_accuracy'])   
    axes[1].plot(history.history['val_categorical_accuracy'])
    axes[1].legend(['categorical_accuracy','val_categorical_accuracy'])

class ModelBuilder():
    
    def initialize_path(self,project_folder):
        self.train_doc = np.random.permutation(open(project_folder + '/' + 'train.csv').readlines())
        self.val_doc = np.random.permutation(open(project_folder + '/' + 'val.csv').readlines())
        self.train_path = project_folder + '/' + 'train'
        self.val_path =  project_folder + '/' + 'val'
        self.num_train_sequences = len(self.train_doc)
        self.num_val_sequences = len(self.val_doc)
        
    def initialize_image_properties(self,image_height=100,image_width=100):
        self.image_height=image_height
        self.image_width=image_width
        self.channels=3
        self.num_classes=27
        self.total_frames=29
          
    def initialize_hyperparams(self,frames_to_sample=30,batch_size=20,num_epochs=20):
        self.frames_to_sample=frames_to_sample
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        
        
    def generator(self,source_path, folder_list, augment=False):
        img_idx = np.round(np.linspace(1,self.total_frames-1,self.frames_to_sample)).astype(int)
        batch_size=self.batch_size
        while True:
            t = np.random.permutation(folder_list)
            num_batches = len(t)//batch_size
        
            for batch in range(num_batches): 
                batch_data, batch_labels= self.one_batch_data(source_path,t,batch,batch_size,img_idx,augment)
                yield batch_data, batch_labels 

            remaining_seq=len(t)%batch_size
        
            if (remaining_seq != 0):
                batch_data, batch_labels= self.one_batch_data(source_path,t,num_batches,batch_size,img_idx,augment,remaining_seq)
                yield batch_data, batch_labels 
    
    
    def one_batch_data(self,source_path,t,batch,batch_size,img_idx,augment,remaining_seq=0):
    
        seq_len = remaining_seq if remaining_seq else batch_size
    
        batch_data = np.zeros((seq_len,len(img_idx),self.image_height,self.image_width,self.channels)) 
        batch_labels = np.zeros((seq_len,self.num_classes)) 
    
        if (augment): batch_data_aug = np.zeros((seq_len,len(img_idx),self.image_height,self.image_width,self.channels))

        
        for folder in range(seq_len): 
            #imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0])
            #imgs=['data_1.jpg', 'data_2.jpg', 'data_3.jpg', 'data_4.jpg', 'data_5.jpg', 'data_6.jpg', 'data_7.jpg', 'data_8.jpg', 'data_9.jpg', 'data_10.jpg', 'data_11.jpg', 'data_12.jpg', 'data_13.jpg', 'data_14.jpg', 'data_15.jpg', 'data_16.jpg', 'data_17.jpg', 'data_18.jpg', 'data_19.jpg', 'data_20.jpg', 'data_21.jpg', 'data_22.jpg', 'data_23.jpg', 'data_24.jpg', 'data_25.jpg', 'data_26.jpg', 'data_27.jpg', 'data_28.jpg', 'data_29.jpg', 'data_30.jpg'] 
            imgs=['result_1.jpg', 'result_2.jpg', 'result_3.jpg', 'result_4.jpg', 'result_5.jpg', 'result_6.jpg', 'result_7.jpg', 'result_8.jpg', 'result_9.jpg', 'result_10.jpg', 'result_11.jpg', 'result_12.jpg', 'result_13.jpg', 'result_14.jpg', 'result_15.jpg', 'result_16.jpg', 'result_17.jpg', 'result_18.jpg', 'result_19.jpg', 'result_20.jpg', 'result_21.jpg', 'result_22.jpg', 'result_23.jpg', 'result_24.jpg', 'result_25.jpg', 'result_26.jpg', 'result_27.jpg', 'result_28.jpg', 'result_29.jpg']
            for idx,item in enumerate(img_idx):
                #print("idx,item: ",idx,item)
                image = imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                image_resized=imresize(image,(self.image_height,self.image_width,3))
                batch_data[folder,idx,:,:,0] = (image_resized[:,:,0])/255
                batch_data[folder,idx,:,:,1] = (image_resized[:,:,1])/255
                batch_data[folder,idx,:,:,2] = (image_resized[:,:,2])/255
                if (augment):
                    shifted = cv2.warpAffine(image, 
                                             np.float32([[1, 0, np.random.randint(-30,30)],[0, 1, np.random.randint(-30,30)]]), 
                                            (image.shape[1], image.shape[0]))
                    
                    gray = cv2.cvtColor(shifted,cv2.COLOR_BGR2GRAY)

                    x0, y0 = np.argwhere(gray > 0).min(axis=0)
                    x1, y1 = np.argwhere(gray > 0).max(axis=0) 
                    
                    cropped=shifted[x0:x1,y0:y1,:]
                    
                    image_resized=imresize(cropped,(self.image_height,self.image_width,3))
                    
                    #shifted = cv2.warpAffine(image_resized, 
                    #                        np.float32([[1, 0, np.random.randint(-3,3)],[0, 1, np.random.randint(-3,3)]]), 
                    #                        (image_resized.shape[1], image_resized.shape[0]))
            
                    batch_data_aug[folder,idx,:,:,0] = (image_resized[:,:,0])/255
                    batch_data_aug[folder,idx,:,:,1] = (image_resized[:,:,1])/255
                    batch_data_aug[folder,idx,:,:,2] = (image_resized[:,:,2])/255
                
            
            batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1  #video thứ/thuộc class thứ =1 (gán nhãn cho các video, class thứ bao nhiêu thì xem file csv lệnh split phần tử cuối cùng)
            
    
        if (augment):
            batch_data=np.concatenate([batch_data,batch_data_aug])
            batch_labels=np.concatenate([batch_labels,batch_labels])

        
        return(batch_data,batch_labels)
    
    
    def train_model(self, model, augment_data=False):
        train_generator = self.generator(self.train_path, self.train_doc,augment=augment_data)
        val_generator = self.generator(self.val_path, self.val_doc)


        model_name = '/home/linhdt/Desktop/sydat/'+'model_init_13_6_gru'+'/'
    
        if not os.path.exists(model_name):
            os.mkdir(model_name)
        
        filepath = model_name + '{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'


        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        LR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=10)
        callbacks_list = [checkpoint, LR]

        if (self.num_train_sequences%self.batch_size) == 0:
            steps_per_epoch = int(self.num_train_sequences/self.batch_size)
        else:
            steps_per_epoch = (self.num_train_sequences//self.batch_size) + 1

        if (self.num_val_sequences%self.batch_size) == 0:
            validation_steps = int(self.num_val_sequences/self.batch_size)
        else:
            validation_steps = (self.num_val_sequences//self.batch_size) + 1
    
        history=model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=self.num_epochs, verbose=1, 
                            callbacks=callbacks_list, validation_data=val_generator, 
                            validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=26)
        return history

        
    @abc.abstractmethod
    def define_model(self):
        pass

        

# from keras.applications import mobilenet

# mobilenet_transfer = mobilenet.MobileNet(weights='imagenet', include_top=False)

# class RNNCNN_TL2(ModelBuilder):
    
#     def define_model(self,gru_cells=64,dense_neurons=64,dropout=0.25):
        
#         model = Sequential()
#         model.add(TimeDistributed(mobilenet_transfer,input_shape=(self.frames_to_sample,self.image_height,self.image_width,self.channels)))
 
        
#         model.add(TimeDistributed(BatchNormalization()))
#         model.add(TimeDistributed(MaxPooling2D((2, 2))))
#         model.add(TimeDistributed(Flatten()))

#         model.add(GRU(gru_cells))
#         model.add(Dropout(dropout))
        
#         model.add(Dense(dense_neurons,activation='relu'))
#         model.add(Dropout(dropout))
        
#         model.add(Dense(self.num_classes, activation='softmax'))
        
        
#         optimiser = optimizers.Adam()
#         model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#         return model

# rnn_cnn_tl2=RNNCNN_TL2()
# rnn_cnn_tl2.initialize_path(project_folder)
# rnn_cnn_tl2.initialize_image_properties(image_height=120,image_width=120)
# rnn_cnn_tl2.initialize_hyperparams(frames_to_sample=16,batch_size=5,num_epochs=60)
# rnn_cnn_tl2_model=rnn_cnn_tl2.define_model(gru_cells=128,dense_neurons=128,dropout=0.25)
# rnn_cnn_tl2_model.summary()

# print("Total Params:", rnn_cnn_tl2_model.count_params())
# history_model19=rnn_cnn_tl2.train_model(rnn_cnn_tl2_model,augment_data=False)






# from keras.applications import mobilenet

# mobilenet_transfer = mobilenet.MobileNet(weights='imagenet', include_top=False)

# class RNNCNN_TL2(ModelBuilder):
    
#     def define_model(self,gru_cells=64,dense_neurons=64,dropout=0.25):
        
#         model = Sequential()
#         model.add(TimeDistributed(mobilenet_transfer,input_shape=(self.frames_to_sample,self.image_height,self.image_width,self.channels)))
 
        
#         model.add(TimeDistributed(BatchNormalization()))
#         model.add(TimeDistributed(MaxPooling2D((2, 2))))
#         model.add(Dropout(0.2))
#         model.add(TimeDistributed(Flatten()))

#         model.add(GRU(gru_cells))
#         model.add(Dropout(0.3))
        
#         model.add(Dense(dense_neurons,activation='relu'))
#         model.add(Dropout(0.4))
        
#         model.add(Dense(self.num_classes, activation='softmax'))
        
        
#         optimiser = optimizers.Adam()
#         model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#         return model 

#---------------------------------------------------------------------------------
from keras import Sequential
# from keras.layers import CuDNNLSTM

        # optimiser = optimizers.Adam(lr=0.0002)
        # model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        # return model


# rnn_cnn_tl2=RNNCNN_TL2()
# rnn_cnn_tl2.initialize_path(project_folder)
# rnn_cnn_tl2.initialize_image_properties(image_height=256,image_width=256)
# rnn_cnn_tl2.initialize_hyperparams(frames_to_sample=16,batch_size=1,num_epochs=90)
# rnn_cnn_tl2_model=rnn_cnn_tl2.define_model(gru_cells=128,dense_neurons=128,dropout=0.25)
# rnn_cnn_tl2_model.summary()
from keras.applications import mobilenet

mobilenet_transfer = mobilenet.MobileNet(weights='imagenet', include_top=False)

class RNNCNN_TL2(ModelBuilder):
    
    def define_model(self,gru_cells=64,dense_neurons=64,dropout=0.25):
        
        model = Sequential()
        model.add(TimeDistributed(mobilenet_transfer,input_shape=(self.frames_to_sample,self.image_height,self.image_width,self.channels)))
 
        
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Flatten()))

        model.add(GRU(gru_cells))
        model.add(Dropout(dropout))
        
        model.add(Dense(dense_neurons,activation='relu'))
        model.add(Dropout(dropout))
        
        model.add(Dense(self.num_classes, activation='softmax'))
        
        
        optimiser = optimizers.Adam()
        model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model
rnn_cnn_tl2=RNNCNN_TL2()
rnn_cnn_tl2.initialize_path(project_folder)
rnn_cnn_tl2.initialize_image_properties(image_height=128,image_width=128)
rnn_cnn_tl2.initialize_hyperparams(frames_to_sample=16,batch_size=8,num_epochs=200)
rnn_cnn_tl2_model=rnn_cnn_tl2.define_model(gru_cells=128,dense_neurons=128,dropout=0.25)
rnn_cnn_tl2_model.summary()


#history_model19=rnn_cnn_tl2.train_model(rnn_cnn_tl2_model,augment_data=False)

from keras.models import load_model
rnn_cnn_tl2_model = load_model('/home/linhdt/Desktop/sydat/h5/dataSeg_13_6.h5')
rnn_cnn_tl2_model.load_weights('/home/linhdt/Desktop/sydat/h5/dataSeg_13_6.h5')
history= rnn_cnn_tl2.train_model(rnn_cnn_tl2_model, augment_data=False)
