#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
# Copyright:    Yuhan Jiang
# Email:        yuhan.jiang@marquette.edu
# Date:         10/03/2020
# Discriptions :
# Major updata : run Project files.

import os

from keras.optimizers import Adam

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numba
import numpy as np
import pandas as pd
import cv2 as cv#载入OpenCV库

import math
import statistics

import csv,datetime,os,gc
import shutil
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from keras.utils import np_utils
from keras.models import Sequential, Input, Model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten, Lambda
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,AveragePooling2D,Conv2DTranspose
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.utils import multi_gpu_model   #导入keras多GPU函数

#defined function
def array_element_counters(a):
    unique, counts = np.unique(a, return_counts=True)
    counters=dict(zip(unique, counts))
    print(counters)
    return counters
def rotate(img,angle):#旋转功能 逆时针
    h, w =img.shape[0:2]# 获取图像尺寸
    center = (w/2, h/2)# 将图像中心设为旋转中心  @04.21.2019 either 4/2=2 or 5/2=2.5
    M = cv.getRotationMatrix2D(center, angle, 1) #执行旋转
    rotated = cv.warpAffine(img,M,(w,h))
    return rotated# 返回旋转后的图像
def imgprint(img0,windowname="image",height=600,width=800):
    cv.namedWindow(windowname, cv.WINDOW_AUTOSIZE)#
    #cv.resizeWindow(windowname,  width,height)
    cv.imshow(windowname,img0) # 显示图片
    '''
    plt.figure(windowname) # 图像窗口名称
    plt.imshow(img0)
    plt.axis('on') # 关掉坐标轴为 off
    plt.title(windowname) # 图像题目
    plt.show()'''
def newdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + '   Successful')
        return True
    else:
        print(path + '   Exists')
        return False
# function: get mode of a equal col row size array
from scipy import stats
def get_Mode_Array_EqualSize(arr):
    unique, counts = np.unique(arr, return_counts=True)
    index=np.where(counts==max(counts))
    return unique[index[0]][0]

def generate_dataset(qsize=32,glb_img_name ="120A",glb_file_path="F:/ImgLib/",repeat=4,cut=False,gridsize=32):
    psize=qsize
    newdir(glb_file_path+glb_img_name+'/Class_Input')#ok
    newdir(glb_file_path+glb_img_name+'/Class_Output')#ok
    #read data
    print("Pre-Processing,Loading input and output image pair:",glb_img_name)
    # 载入input图像@Ortho-image
    if os.path.exists(glb_file_path+glb_img_name+'/'+glb_img_name+'Ortho_image.jpg'):
        Ortho_image = cv.imread(glb_file_path+glb_img_name+'/'+glb_img_name+'Ortho_image.jpg',1)# 导入图片1/2H
    else:
        print("Ortho_image File is not existing")
    imageH,imageW=Ortho_image.shape[0],Ortho_image.shape[1]#
    print('Original Image Height=',imageH,'Image Width=',imageW)
    #Ortho_image= cv.cvtColor(Ortho_image,cv.COLOR_BGR2GRAY)#转换为灰度图片
    print("Ortho-image cut shape:",cut,Ortho_image.shape)
    # 载入output图像1@Class_image
    if os.path.exists(glb_file_path+glb_img_name+'/'+glb_img_name+'Label_image.csv'): # 2019.09.02
        # read label image 2019.09.02
        PixelP=pd.read_csv(glb_file_path+glb_img_name+'/'+glb_img_name+'Label_image.csv',index_col=False,header=None)
        print('number of Pixels Loaded:',len(PixelP),len(PixelP[1]))
        Class_image=np.array(PixelP).reshape((imageH,imageW)).copy()  # 2019.09.02
        print('CSV_Label_image Successful! shape of Label_image:',Class_image.shape)
        #array_element_counters(Class_image)
    else:
        print("Label_image File is not existing")

    print("Label_image shape,Cut:",cut,Class_image.shape)
    imageH,imageW=Class_image.shape[0],Class_image.shape[1]# update size

     # if 4 rotation 0 90 180 270
    num=(2*int(imageH/qsize)-1)*(2*int(imageW/psize)-1)*repeat
    print("num",num)
    X=np.zeros(shape=(num,qsize,psize,3)) #RGB
    Y=np.zeros(shape=(num,qsize,psize,1))
    C=np.zeros(shape=(num,1)) # num col 1 row

    #roation and repeate
    i=0
    for j in range(0,repeat):
        Ortho_image=rotate(Ortho_image,90*j)
        Class_image=rotate(Class_image,90*j)
        data_input=np.zeros((qsize,psize,3))
        data_output=np.zeros((qsize,psize,1))
        m=0
        while m<=imageH-qsize:
            n=0
            while n<=imageW-psize:
                data_input=Ortho_image[m:m+qsize,n:n+psize,:].copy()
                data_output=Class_image[m:m+qsize,n:n+psize].copy()
                X[i,:]=data_input.reshape((1,qsize,psize,3))# qsize*qsize grayscale 2D image
                Y[i,:]=data_output.reshape((1,qsize,psize,1))# need to be 1 number. 2019/8/25
                C[i]=get_Mode_Array_EqualSize(Y[i,:]) # used the model as the value
                #cv.imwrite(glb_file_path+"/"+glb_img_name+'/Input/'+glb_img_name+'input_'+str(i)+'.jpg',data_input,[int( cv.IMWRITE_JPEG_QUALITY), 100])
                #cv.imwrite(glb_file_path+"/"+glb_img_name+'/Output/'+glb_img_name+'output_'+str(i)+'.jpg',data_output,[int(cv.IMWRITE_JPEG_QUALITY),100])
                i=i+1
                n=n+int(psize/2)
            m=m+int(qsize/2)
        print("Generated dataset number",str(i))
    return X,Y,C,qsize,imageH,imageW

from keras.utils.np_utils import to_categorical
from keras import backend as K
def dice_coef(y_true, y_pred, smooth=0.9):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def IOU_calc(y_true, y_pred, smooth=0.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)
def ClassficationV1_1(x_train,y_train,qsize,n_epochs,batch_size,X_test,Y_test,early_stop=5,valid_split=0.5):#add ealry_stop 2019.09.07 #50 accuracny
    psize=qsize
    x_train=(x_train/255)
    input_shape=x_train[0].shape
    n_classes= 6  # len(set(Y_train))
    y_train=to_categorical(y_train,n_classes)

    X_test=(X_test/255)
    Y_test=to_categorical(Y_test,n_classes)

    print('Patch[,],Dataset,epochs,batch,valid_split,early',qsize,psize,x_train.shape,n_epochs,batch_size,valid_split,early_stop)


    model=Sequential()
    model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=input_shape))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    model.add(Dropout(0.5))#@2019.09.07
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(512,activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(128,activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes,activation='softmax'))

    opt=Adam(1e-4)

    model.summary()  # Output summary of network
    model=multi_gpu_model(model)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['categorical_accuracy',IOU_calc,'accuracy'])
    callbacks=[EarlyStopping(monitor='val_loss',patience=early_stop)]# 0 no 1 bar 2 line
    history=model.fit(x_train,y_train,batch_size=batch_size,epochs=n_epochs,verbose=2,validation_split=valid_split,callbacks=callbacks)

    score=model.evaluate(X_test,Y_test,verbose=2)
    print('Test loss:',score[0])
    print('Test accuracy:',score[1])

    preds=model.predict(X_test)# Extract predictions

    n_examples=10
    plt.figure(figsize=(15,5),dpi=600)
    for i in range(n_examples):
        ax=plt.subplot(2,n_examples,i+1)
        plt.imshow(X_test[i,:,:,0], vmin=-0.5,vmax=9.5,cmap='tab10')
        plt.title("Label: {}\nPredicted: {}".format(np.argmax(Y_test[i]),np.argmax(preds[i])))
        plt.axis('off')

    plt.savefig(glb_file_path+'Class Model_'+str(Split)+'_Predication.svg')
    print("Predication saved")

    plt.figure(figsize=(15,5),dpi=600)
    j=1
    for i in range(len(Y_test)):
        if (j>10):
            break
        label=np.argmax(Y_test[i])
        pred=np.argmax(preds[i])
        if label!=pred:
            ax=plt.subplot(2,n_examples,j)
            plt.imshow(X_test[i,:,:,0], vmin=-0.5,vmax=9.5,cmap='tab10')
            plt.title("Label: {}\nPredicted: {}".format(label,pred))
            plt.axis('off')
            j+=1
    #plt.show()
    plt.savefig(glb_file_path+'Class Model_'+str(Split)+'_PredError.svg')
    print("Predication Error saved")
    #plt.show()
    plt.savefig(glb_file_path+'Class Model_'+str(Split)+'_Predication.svg')
    print("Predication saved")

    del x_train,y_train
    gc.collect()
    plt.close(fig='all')
    return model,history

# draw the loss
def printloss(history,glb_file_path="F:/ImgLib/",qsize=32):
    #print(history.history.keys())
    los=history.history['loss']
    val_los=history.history['val_loss']
    plt.figure(figsize=(10,5),dpi=600)
    plt.plot(np.arange(len(los)),los,label='training')
    plt.plot(np.arange(len(val_los)),val_los,label='validation')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc=0)
    plt.savefig(glb_file_path+'_CNN_'+str(Split)+'_loss.svg')
    print("loss_history saved")
    plt.close(fig='all')
    #plt.show()#//8/13
    data=pd.DataFrame(history.history)  #为了能够使这组数据成为可以让pandas处理的数据，需要通过这个数组创建DataFrame。
    data.to_csv(glb_file_path+'_CNN_'+str(Split)+'_loss.csv',index=True,header=True)

# draw the accuaracy
def printacc(history,glb_file_path="F:/ImgLib/",qsize=32):
    #print(history.history.keys())
    acc=history.history['accuracy']#'dice_coef',
    val_acc=history.history['val_accuracy']#'val_dice_coef',
    plt.figure(figsize=(10,5),dpi=600)
    plt.plot(np.arange(len(acc)),acc,label='training')
    plt.plot(np.arange(len(val_acc)),val_acc,label='validation')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc=0)
    plt.savefig(glb_file_path+'_CNN_'+str(Split)+'_acc.svg')
    print("acc_history saved")
    plt.close(fig='all')
    #plt.show()#//8/13
    data=pd.DataFrame(history.history)  #为了能够使这组数据成为可以让pandas处理的数据，需要通过这个数组创建DataFrame。
    data.to_csv(glb_file_path+'_CNN_'+str(Split)+'_acc.csv',index=True,header=True)

# draw the IOU
def printIoU(history,glb_file_path="F:/ImgLib/",qsize=32):
    #print(history.history.keys())
    acc=history.history['IOU_calc']#'dice_coef',
    val_acc=history.history['val_IOU_calc']#'val_dice_coef',
    plt.figure(figsize=(10,5),dpi=600)
    plt.plot(np.arange(len(acc)),acc,label='training')
    plt.plot(np.arange(len(val_acc)),val_acc,label='validation')
    plt.title('IoU_calc')
    plt.xlabel('epochs')
    plt.ylabel('IoU')
    plt.legend(loc=0)
    plt.savefig(glb_file_path+'_CNN_'+str(Split)+'_IoU.svg')
    print("IoU_history saved")
    plt.close(fig='all')
    #plt.show()#//8/13
    data=pd.DataFrame(history.history)  #为了能够使这组数据成为可以让pandas处理的数据，需要通过这个数组创建DataFrame。
    data.to_csv(glb_file_path+'_CNN_'+str(Split)+'_IoU.csv',index=True,header=True)

def Evaluation(deepele,qsize=32,glb_img_name ="120A",glb_file_path="F:/ImgLib/",Project_bool=False):
    psize=qsize
    newdir(glb_file_path+glb_img_name+'/Class_Input')
    newdir(glb_file_path+glb_img_name+'/Class_Output')
    if Project_bool:
        shutil.copy(glb_file_path+'/'+glb_img_name+'Ortho_image.jpg',glb_file_path+glb_img_name+'/')
    #read data
    # 载入input图像@EHENCED_Class_image
    if os.path.exists(glb_file_path+glb_img_name+'/'+glb_img_name+'Ortho_image.jpg'):
        Ortho_image = cv.imread(glb_file_path+glb_img_name+'/'+glb_img_name+'Ortho_image.jpg',1)# 导入图片1/2H
    else:
        print("Ortho_image File is not existing")
    if os.path.exists(glb_file_path+glb_img_name+'/'+glb_img_name+'Label_image.csv'):#2019.09.02
        # read label image 2019.09.02
        PixelP=pd.read_csv(glb_file_path+glb_img_name+'/'+glb_img_name+'Label_image.csv',index_col=False,header=None)
        print('number of Pixels Loaded:',len(PixelP),len(PixelP[1]))
        Class_image=np.array(PixelP).reshape(Ortho_image.shape[0:2]).copy()  # 2019.09.02
        print('CSV_Label_image Successful! shape of Label_image:',Class_image.shape)
    elif os.path.exists(glb_file_path+glb_img_name+'/'+glb_img_name+'Label_image.jpg'):
        Class_image = cv.imread(glb_file_path+glb_img_name+'/'+glb_img_name+'Label_image.jpg',1)# 导入图片1/2H
        Class_image=cv.cvtColor(Class_image,cv.COLOR_BGR2GRAY)  #转换为恢复图片
        print('JPG_Label_image Successful!')
    else:
        print("Label_image File is not existing")
        Class_image=Ortho_image.copy()
        Class_image=cv.cvtColor(Class_image,cv.COLOR_BGR2GRAY)  #转换为恢复图片
    # color label
    if os.path.exists(glb_file_path+'Public_Object_Label_Dic.csv'):
        colordic={}
        with open(glb_file_path+'Public_Object_Label_Dic.csv') as f:
            colordic=dict(filter(None,csv.reader(f)))
            for key,val in colordic.items():
                colordic[key]=int(val)
        print(colordic)
        colorlist=list(map(int,list(colordic.values())))
        colorLabel=list(map(str,list(colordic.keys())))
    else:
        print("Object Label Dic is not existing")
        colordic={'Default':0, 'Pavement/Line Mark/Bridge/OtherPavementSurface':1,'Truck/Bus/Car':2,'Light/TrafficSign':3,'Crack':4,'VegetationZoo':5}
        print(colordic)
        colordic={'d':0, 'p':1,'t':2,'l':3,'c':4,'v':5}
        print(colordic)
        print("Created New Object Label Dic")
        colorlist=list(map(int,list(colordic.values())))
        colorLabel=list(map(str,list(colordic.keys())))

    imageH,imageW=Ortho_image.shape[0],Ortho_image.shape[1]#

    color_o_image=cv.cvtColor(Ortho_image,cv.COLOR_BGR2RGB)
    #Ortho_image= cv.cvtColor(Ortho_image,cv.COLOR_BGR2GRAY)#转换为恢复图片


    num=(2*int(imageH/qsize)-1)*(2*int(imageW/psize)-1)    #print("num",num)
    X=np.zeros(shape=(num,qsize,psize,3))
    A=np.zeros(shape=(num,qsize,psize,3))
    C=[]#reocord ortho-image partial
    E=[]#reocord Elevation-map partial

    #roation and repeate
    i=0
    data_input=np.zeros((qsize,psize,3))
    m=0
    overlap=1
    while m<=imageH-qsize:
            n=0
            while n<=imageW-psize:
                data_input=Ortho_image[m:m+qsize,n:n+psize]
                X[i,:]=data_input.reshape((1,qsize,psize,3))
                i=i+1
                if overlap:n=n+int(psize/2)
                else: n=n+psize
            if overlap:m=m+int(qsize/2)
            else: m=m+qsize

    i=0
    data_input=np.zeros((qsize,psize,3))
    m=0
    overlap=0
    while m<=imageH-qsize:
            n=0
            while n<=imageW-psize:
                data_input=Ortho_image[m:m+qsize,n:n+psize]
                A[i,:]=data_input.reshape((1,qsize,psize,3))
                C.append(color_o_image[m:m+qsize,n:n+psize])
                E.append(Class_image[m:m+qsize,n:n+psize])
                i=i+1
                if overlap:n=n+int(psize/2)
                else: n=n+psize
            if overlap:m=m+int(qsize/2)
            else: m=m+qsize

    pred_Patch_Class=deepele.predict(A/255) # prediction result
    pred_overlap=deepele.predict(X/255)
    pred_Assembly=Ortho_image[:,:,0].copy()
    pred_Assembly[:]=0#拼接结果
    i=0
    m=0
    while m<=imageH-qsize:
        n=0
        while n<=imageW-psize:  #m row; n col
            patchClass=np.zeros((qsize,psize))
            patchClass[:]=int(np.argmax(pred_Patch_Class[i]))# [0,6] use one max prob.
            pred_Assembly[m:m+qsize,n:n+psize]=patchClass
            i=i+1
            n=n+psize
        m=m+qsize
    print('sidebyside_Full From Pics',i)
    #imgprint(pred_Assembly/255,glb_img_name+'unoverlap')
    SideBySide=pred_Assembly.copy()# save as sidebyside result
    array_element_counters(pred_Assembly)
    #update center part; overlaping
    i=0
    m=0
    while m<=imageH-qsize:
        n=0
        while n<=imageW-psize:  #m row; n col
            overlap_patchClass=np.zeros((qsize,psize))
            overlap_patchClass[:]=int(np.argmax(pred_overlap[i]))
            pred_Assembly[m+int(qsize*.25):m+int(qsize*.75),n+int(psize*.25):n+int(psize*.75)]=overlap_patchClass.reshape(int(qsize),int(psize))[int(qsize*0.25):int(qsize*0.75),int(psize*0.25):int(psize*0.75)]  #
            #edge
            if m==0 and (n!=0 and n!=imageW-psize):#upper edge
                pred_Assembly[m:m+int(qsize*.25),n+int(psize*.25):n+int(psize*.75)]=overlap_patchClass.reshape(int(qsize),int(psize))[:int(qsize*0.25),int(psize*0.25):int(psize*0.75)]  #
            elif m==imageH-qsize and (n!=0 and n!=imageW-psize):# bottom edge
                pred_Assembly[m+int(qsize*.75):m+qsize,n+int(psize*.25):n+int(psize*.75)]=overlap_patchClass.reshape(int(qsize),int(psize))[int(qsize*0.75):,int(psize*0.25):int(psize*0.75)]  #
            elif n==0 and (m!=0 and m!=imageH-qsize):#left edge
                pred_Assembly[m+int(qsize*.25):m+int(qsize*.75),n:n+int(psize*.25)]=overlap_patchClass.reshape(int(qsize),int(psize))[int(qsize*0.25):int(qsize*0.75),:int(psize*0.25)]  #
            elif n==imageW-psize and (m!=0 and m!=imageH-qsize):#right edge
                pred_Assembly[m+int(qsize*.25):m+int(qsize*.75),n+int(psize*.75):n+psize]=overlap_patchClass.reshape(int(qsize),int(psize))[int(qsize*0.25):int(qsize*0.75),int(psize*0.75):]  #
            i=i+1
            n=n+int(psize/2)
        m=m+int(qsize/2)
    print('overlap_Full From Pics',i)
    array_element_counters(pred_Assembly)
    #imgprint(pred_Assembly/255,glb_img_name+'overlap')

    cv.imwrite(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+str(Split)+'_Evaluation_Pred_Class.jpg',pred_Assembly,[int(cv.IMWRITE_JPEG_QUALITY),100])
    #save label image 2019.09.02
    data=pd.DataFrame(pred_Assembly)  #pandas DataFrame。
    data.to_csv(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+str(Split)+'_Evaluation_Pred_Class.csv',index=False,header=False)  #2019.09.02

    kl=max(2,2*imageW//psize)
    i=0
    n=50
    plt.figure(figsize=(n,5),dpi=600)
    for i in range(n):
        # plot original
        ax=plt.subplot(3,n,i+1)
        plt.imshow(C[i+kl])
        ax=plt.subplot(3,n,i+1+n*1)
        plt.imshow(E[i+kl],vmin=-0.5,vmax=9.5,cmap='tab10')
        ax=plt.subplot(3,n,i+1+n*2)
        ptC=np.zeros((qsize,psize))
        ptC[:]=int(np.argmax(pred_Patch_Class[i+kl]))
        plt.imshow(ptC,vmin=-0.5,vmax=9.5,cmap='tab10')
    plt.tight_layout()
    #plt.show()
    plt.savefig(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+'Class Model Trainning_'+str(Split)+'_Sample.svg')

    plt.figure(figsize=(15,12),dpi=600)
    ax1=plt.subplot(2,2,1)
    im1=plt.imshow(SideBySide,vmin=-0.5,vmax=9.5,cmap='tab10')
    ax1.set_xlabel("Side-by-Side")
    ax2=plt.subplot(2,2,2)
    im2=plt.imshow(pred_Assembly,vmin=-0.5,vmax=9.5,cmap='tab10')
    ax2.set_xlabel("Overlapping")
    ax3=plt.subplot(2,2,3)
    im3=plt.imshow(Class_image,vmin=-0.5,vmax=9.5,cmap='tab10')
    ax3.set_xlabel("Label_image")
    ax4=plt.subplot(2,2,4)
    plt.imshow(color_o_image)
    ax4.set_xlabel("Ortho_image")

    # Colorbar
    divider1=make_axes_locatable(ax1)#ax1
    cax1=divider1.append_axes("right",size="1.5%",pad=0.05)  #colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
    cbar1=plt.colorbar(im1,cax=cax1,cmap='tab10')
    cbar1.set_ticks(colorlist)  # color bar
    cbar1.ax.get_yaxis().set_ticks([])
    for j,lab in enumerate(colorLabel):
        cbar1.ax.text(10,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center')  #                         colorlist[0],colorLabel[0])

    divider2=make_axes_locatable(ax2)#ax2
    cax2=divider2.append_axes("right",size="1.5%",pad=0.05)  #colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
    cbar2=plt.colorbar(im2,cax=cax2,cmap='tab10')
    cbar2.set_ticks(colorlist)  # color bar
    cbar2.ax.get_yaxis().set_ticks([])
    for j,lab in enumerate(colorLabel):
        cbar2.ax.text(10,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center')  #                         colorlist[0],colorLabel[0])

    divider3=make_axes_locatable(ax3)#ax3
    cax3=divider3.append_axes("right",size="1.5%",pad=0.05)  #colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
    cbar3=plt.colorbar(im3,cax=cax3,cmap='tab10')
    cbar3.set_ticks(colorlist)  # color bar
    cbar3.ax.get_yaxis().set_ticks([])
    for j,lab in enumerate(colorLabel):
        cbar3.ax.text(10,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center')  #                         colorlist[0],colorLabel[0])
    plt.tight_layout()
    #plt.show()
    plt.savefig(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+'Class Model Trainning_'+str(Split)+'_Comparing.svg')
    del X,A,C,E
    plt.close(fig='all')

def repeatmodeloutput(qsize,n_epochs,batch_size,glb_file_path,early_stop,Split=0.5,ProjectList=None,glb_project_path=None):
    RepNum=4
    psize=qsize
    GSize=min(qsize,psize)//2

    TrainingList=['0','1','2','3','4','6']#,'12','13','14']
    X_train=[]#None,None,None,None,None,None,None,None,None]
    Y_train=[]#None,None,None,None,None,None,None,None,None]
    C_train=[]
    for i in range(len(TrainingList)):
        X_input_c,Y_input_c, C_input_c =generate_dataset(qsize,glb_img_name=TrainingList[i],glb_file_path=glb_file_path,repeat=RepNum,gridsize=GSize)[0:3]
        X_train.append(X_input_c)
        del X_input_c
        gc.collect()
        Y_train.append(Y_input_c)
        del Y_input_c
        gc.collect()
        C_train.append(C_input_c)
        del C_input_c
        gc.collect()

    X_train=np.vstack(X_train)#,X4
    Y_train=np.vstack(Y_train)#,Y4
    C_train=np.vstack(C_train)  #,C4

    print(X_train.shape,Y_train.shape,C_train.shape)

    X_t,Y_t,C_t=generate_dataset(qsize,glb_img_name='1',glb_file_path=glb_file_path,repeat=1,gridsize=GSize)[0:3]

    deepelev1,historyplot =ClassficationV1_1(X_train,C_train,qsize,n_epochs,batch_size,X_t,C_t,early_stop=early_stop,valid_split=Split) # neetwork self normalazion ClassficationV1(X_train,Y_train,qsize,n_epochs,batch_size,X_test,Y_test)

    del X_train,Y_train,C_train
    gc.collect()

    printloss(historyplot,glb_file_path=glb_file_path,qsize=qsize)
    printacc(historyplot,glb_file_path=glb_file_path,qsize=qsize)
    printIoU(historyplot,glb_file_path=glb_file_path,qsize=qsize)
    #'''
    print("=====Split:",Split,"============================================================================================================Evaluation")
    plt.close(fig='all')
    for i in TrainingList:
        Evaluation(deepelev1,qsize,glb_img_name=str(i),glb_file_path=glb_file_path)
    plt.close(fig='all')
    if ProjectList!=None:
        print("=====Split:",Split,"============================================================================================================Project")
        for i in ProjectList:
            Evaluation(deepelev1,qsize,glb_img_name=str(i),glb_file_path=glb_project_path,Project_bool=True)
    #'''
    del deepelev1
    gc.collect()

#=============================================================================
if __name__ == '__main__':
    #Running Parmaters   1212
    #-----------------------------
    if os.path.exists('D:/'):
        glb_file_path='D:/CentOS/G_Training/'
        glb_project_path='D:/CentOS/G2/'
        b='win'
        cpu=3
    elif os.path.exists('/data/'):
        glb_file_path='/data/G_Training/'
        glb_project_path='/data/G3/'
        b='server'
        cpu=4
    print(glb_file_path,end=' ')
    qsize=16# best 16
    n_epochs=100  # 1000
    early_stop=10

    batch_size=512
    Split=.5
    plt.close(fig='all')
    print("=====Split:",Split,"=====================================================================")
    ProjectList=list(range(53))# 37 is the number of image in the project file for eveluation
    repeatmodeloutput(qsize,n_epochs,batch_size,glb_file_path,early_stop,Split=Split,ProjectList=ProjectList,glb_project_path=glb_project_path)
    gc.collect()