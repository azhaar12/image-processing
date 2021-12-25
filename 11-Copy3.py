#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import nibabel as nib
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import to_categorical
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras import backend as K

import cv2 as cv2

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# In[2]:


def my_load_data(path, NoOfFiles):
    """
    param path: path of the training dataset
    returns:
        data: files of type flair, t1. t1_ce and t2
        gt: segmented tumor in the file types
    """
    
#     path = path + 'Brats18_CBICA_AAM_'
#     my_dir = sorted(os.listdir(path))
    my_dir = sorted(os.listdir(path))
#     print(my_dir)
    data = []
    myFolders = my_dir[1:NoOfFiles+1]
    # Taqadum Guage
    pbar = tqdm_notebook(total=100)
    k=0
    for Folder in myFolders:
        k=k+1
        pbar.update(1)
        myPath = path + Folder
        segFile = myPath+ '/'+Folder+'_seg.nii.gz'
#        print("Reading File: ", myPath)
        flairFile = myPath+ '/'+Folder+'_flair.nii.gz'
#        t1File = myPath+ '/'+Folder+'_t1.nii.gz'
#         t1ceFile = myPath+ '/'+Folder+'_t1ce.nii'
#        t2File = myPath+ '/'+Folder+'_t2.'
        
        seg = np.array(nib.load(segFile).get_fdata())
        
        flair = np.array(nib.load(flairFile).get_fdata())

#        t1 = np.array(nib.load(t1File).get_fdata())

#         t1ce = np.array(nib.load(t1ceFile).get_fdata())

#         t2 = np.array(nib.load(t2File).get_fdata())

#         data.append([flair, t1, t1ce, t2, seg])
        data.append([flair, seg])
    data = np.array(data)
    kk = round(100-k)/3
    pbar.update(kk)
    kk=kk+5
    data = np.rint(data).astype(np.int16)
    pbar.update(kk)
    data = data[:, :, :, :]
    pbar.update(round(kk-10))
    data = np.transpose(data)
    pbar.close()
    return data


# In[3]:


#path= '../input/brats2018/MICCAI_BraTS_2018_Data_Training/'
path = 'C:/Users/venous/Desktop/MICCAI_BraTS_2018_Data_Training-1/HGG/'
data = my_load_data(path,20)


# In[4]:


data.shape, data.dtype


# In[5]:


data = np.transpose(data, (4,0,1,2,3))
print(data.shape)


# In[6]:


fig = plt.figure(figsize=(5,5))
immmg = data[4][100,:,:,0]
imgplot = plt.imshow(immmg, 'gray')
plt.show()


# In[7]:


def Data_Concatenate(input_data):
    counter = 0
    output = []
    for i in range(2):
        print('$')
        c=0; counter=0;
        for ii in range(len(input_data)):
            if (counter < len(input_data)):
                a = input_data[counter][:,:,:,i]
                b = input_data[counter+1][:,:,:,i]
                
                if (counter == 0):
                    c = np.concatenate((a,b), axis=0)
                    print('c1={}'.format(c.shape))
                    counter += 2
                else:
                    c1 = np.concatenate((a,b), axis=0)
                    c = np.concatenate((c,c1), axis=0)
                    print('c2={}'.format(c.shape))
                    counter += 2
        c = c[:,:,:,np.newaxis]
        output.append(c)
    return output


# In[8]:


indata = Data_Concatenate(data)


# In[9]:


AIO = concatenate(indata, axis=3)
AIO = np.array(AIO, dtype=np.float32)
TR = np.array(AIO[:,:,:,0], dtype=np.float32) # Training
TRL = np.array(AIO[:,:,:,1], dtype=np.float32) # Fro test use the segmentation


# In[10]:


X_train, X_test, Y_train, Y_test = train_test_split(TR, TRL, test_size=0.15, random_state=32)
AIO=0
print("X Train Shape: ", X_train.shape, "X train Type: ", X_train.dtype)
print("Y Train Shape: " ,Y_train.shape, " Y train Type: ", Y_test.dtype)


# In[11]:


fig = plt.figure(figsize=(15,8))
ax1 = fig.add_subplot(121)
ax1.imshow(X_train[1],'gray')

ax2 = fig.add_subplot(122)
ax2.imshow(Y_train[1],'gray')


# In[12]:


# Converting original image to Stationary wavelet transformed image
from pywt import swt2
X_trainWT = X_train
X_testWT = X_test
for i in range(len(X_train)):
    c = swt2(data=X_train[i],wavelet='db1',level=1)
    X_trainWT[i] = c[0][0]
    c=0

for i in range(len(X_test)):
    c = swt2(data=X_test[i], wavelet='db1',level=1)
    X_testWT[i] = c[0][0]
    c=0


# In[13]:


print("X Train Shape: ", X_train.shape, "X train Type: ", X_train.dtype)
print("After WT ::: X Train Shape : ", X_trainWT.shape, "X train Type: ", X_train.dtype)
print("No WT applied on Y:::Y Train Shape: " ,Y_train.shape, " Y train Type: ", Y_test.dtype)


# In[14]:


fig = plt.figure(figsize=(15,8))
ax1 = fig.add_subplot(121)
ax1.imshow(X_train[1],'gray')
ax1.title.set_text("X train Before WT")

ax2 = fig.add_subplot(122)
ax2.imshow(X_trainWT[1],'gray')
ax2.title.set_text("X train After WT")


# In[15]:


normalImage = (X_trainWT[1]/np.max(X_trainWT[1]))
print("Range of image after WT = ", np.min(X_trainWT[1]), " to ", np.max(X_trainWT[1]))
print("Range of image after Normalization = ", np.min(normalImage), " to ", np.max(normalImage))
fig = plt.figure(figsize=(15,8))
ax1 = fig.add_subplot(121)
ax1.imshow(X_trainWT[1],'gray')
ax1.title.set_text("X train Afeter WT Before normalizatio")

ax2 = fig.add_subplot(122)
ax2.imshow(normalImage,'gray')
ax2.title.set_text("X train After WT and after normalization" )


# In[16]:


#X_trainWTNorm = X_trainWT
#for i in range(0,X_trainWT.shape[0]):
#    X_trainWTNorm[i,:,:] = (X_trainWT[i]/np.max(X_trainWT[i]))

#print("Range of image after Normalization = ", np.min(X_trainWTNorm[10]), " to ", np.max(X_trainWTNorm[10]))
#print("Range of image Before Normalization = ", np.min(X_trainWT[10]), " to ", np.max(X_trainWT[10]))
#X_trainWTNorm[100,:,:]
    


# In[17]:


#X_testWTNorm = X_testWT
#for i in range(0,X_testWT.shape[0]):
    #X_testWTNorm[i,:,:] = (X_testWT[i]/np.max(X_testWT[i]))

#print("Range of image after Normalization = ", np.min(X_testWTNorm[10]), " to ", np.max(X_testWTNorm[10]))
#print("Range of image Before Normalization = ", np.min(X_testWT[10]), " to ", np.max(X_testWT[10]))
#X_testWTNorm.shape


# In[18]:


#Y_trainNorm = Y_train
#for i in range(0,Y_train.shape[0]):
    #Y_trainNorm[i,:,:] = (Y_train[i]/np.max(Y_train[i]))

#print("Range of image after Normalization = ", np.min(Y_trainNorm[10]), " to ", np.max(Y_trainNorm[10]))
#print("Range of image Before Normalization = ", np.min(Y_train[10]), " to ", np.max(Y_train[10]))
#X_testWTNorm.shape


# ## U-Net Model Implementation

# In[19]:


def Convolution(input_tensor, filters):
    
    x = Conv2D(filters=filters, kernel_size=(3,3), padding='same', strides=(1,1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def model(input_shape):
    
    inputs = Input((input_shape))
    
    conv_1 = Convolution(inputs, 32)
    maxp_1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv_1)
    
    conv_2 = Convolution(maxp_1, 64)
    maxp_2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv_2)
    
    conv_3 = Convolution(maxp_2, 128)
    maxp_3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv_3)
    
    conv_4 = Convolution(maxp_3, 256)
    maxp_4 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv_4)
    
    conv_5 = Convolution(maxp_4, 512)
    upsample_6 = UpSampling2D((2,2))(conv_5)
    
    conv_6 = Convolution(upsample_6, 256)
    upsample_7 = UpSampling2D((2,2))(conv_6)
    
    upsample_7 = concatenate([upsample_7, conv_3])
    
    conv_7 = Convolution(upsample_7, 128)
    upsample_8 = UpSampling2D((2,2))(conv_7)
    
    conv_8 = Convolution(upsample_8, 64)
    upsample_9 = UpSampling2D((2,2))(conv_8)
    
    upsample_9 = concatenate([upsample_9, conv_1])
    
    conv_9 = Convolution(upsample_9, 32)
    outputs = Conv2D(1, (1,1), activation='sigmoid')(conv_9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


# In[20]:


# Loading the Light weighted CNN
model = model(input_shape=(240,240,1))
#model.summary()


# In[21]:


model.summary()


# In[22]:


# Computing Dice_Coefficient
def dice_coef(y_true, y_pred, smooth=1):
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # Normalization
    y_true_f = y_true_f/K.max(y_true_f)
    y_pred_f = y_pred_f/K.max(y_pred_f)
    
    intersection = K.sum(y_true_f * y_pred_f)
   # return (2. * intersection ) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))*1.42
   #y_true_f = K.flatten(y_true)
   # y_pred_f = K.flatten(y_pred)
   #intersection = K.sum(y_true_f * y_pred_f)
   # return (2. * intersection ) / (K.sum(y_true_f) + K.sum(y_pred_f))
# Computing Precision
def precision(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    
    return precision

# Computing Sensitivity
def sensitivity(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    return true_positives / (possible_positives + K.epsilon())

# Computing Specificity
def specificity(y_true, y_pred):
    
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    
    return true_negatives / (possible_negatives + K.epsilon())


# In[23]:


# Compiling the model
Adam = optimizers.Adam(lr=0.001)
model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['accuracy', dice_coef, precision, sensitivity, specificity])


# In[24]:


K.min(X_trainWT[1])


# In[25]:


# Fitting the model over the data

history = model.fit(X_train, Y_train, batch_size=32, epochs=25, validation_split=0.20,verbose=1,initial_epoch=0)


# In[ ]:


# Evaluating the model on the training and testing data
model.evaluate(x=X_train, y=Y_train, batch_size=32, verbose=1, sample_weight=None, steps=None)
model.evaluate(x=X_test, y=Y_test, batch_size=32, verbose=1, sample_weight=None, steps=None)


# In[ ]:


# Accuracy vs Epoch
def Accuracy_Graph(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    #plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()
    
# Dice Similarity Coefficient vs Epoch
def Dice_coefficient_Graph(history):

    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    #plt.title('Dice_Coefficient')
    plt.ylabel('Dice_Coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()
    
# Precision vs Epoch
def Precision_Graph(history):

    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    #plt.title('Dice_Coefficient')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower left')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()

# Loss vs Epoch
def Loss_Graph(history):

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()


# In[ ]:


# Plotting the Graphs of Accuracy, Dice_coefficient, Loss at each epoch on Training and Testing data
Accuracy_Graph(history)
Dice_coefficient_Graph(history)
Loss_Graph(history)


# In[ ]:


model.save('./BraTs2020_swt_db1_l1.h5')


# In[ ]:


model.load_weights('./BraTs2020_swt_db1_l1.h5')


# In[ ]:


X_train=X_test=Y_train=Y_test=0


# In[ ]:


fig = plt.figure(figsize=(5,5))
immmg = TR[210,:,:]
imgplot = plt.imshow(immmg)
plt.show()


# In[ ]:


from pywt import swt2
for i in range(len(TR)):
    c = swt2(data=TR[i],wavelet='db1',level=1)
    TR[i] = c[0][0]
    c=0


# In[ ]:


pref_tumor = model.predict(TR)


# In[ ]:


ind = 5 # From 1 to 20 Index of the omage
sliceNo = 90 # From 1 to 155
a=(ind-1)*155+sliceNo
plt.figure(figsize=(15,10))
plt.subplot(132)
plt.title('Detacted by model')
plt.axis('off')
plt.imshow(np.squeeze(TR[a,:,:]),cmap='gray')
plt.imshow(np.squeeze(pref_tumor[a,:,:]),alpha=0.3,cmap='Reds')

plt.subplot(131)
plt.title('Original MRI')
plt.axis('off')
plt.imshow(np.squeeze(TR[a,:,:]),cmap='gray')

maxVal = np.max(TRL[a,:,:])
for i in range (0,240):
    for j in range(0,240):
        if(TRL[a,i,j]>0):
            TRL[a,i,j]=maxVal

plt.subplot(133)
plt.title('Detected by radiolgest')
plt.axis('off')

plt.imshow(np.squeeze(TRL[a,:,:]),cmap='gray')


# ###### 

# In[ ]:




