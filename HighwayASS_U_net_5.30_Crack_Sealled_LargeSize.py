#!/usr/bin/env python3.7.3
# -*- coding: utf-8 -*-
# Copyright:    Yuhan Jiang
# Email:        yuhan.jiang@marquette.edu
# Date:         5/30/2020
# Discriptions : generate dataset from a ortho-image and Class-image pair
# Major updata : inintial version of image segementation;'Public_Object_Label_Dic.csv'; array mode update; change dirc.
#
# Expecting    : add to CNN_elevation estimation
import os

from keras.utils import multi_gpu_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numba
import numpy as np
import pandas as pd
import cv2 as cv#载入OpenCV库

import math
import statistics

import csv,datetime,os,gc

from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import glob

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,EarlyStopping
from keras import backend as K
from keras.models import Model
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    # 引入模块
    import os
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + '   Successful')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + '   Exists')
        return False
# function: get mode of a equal col row size array
from scipy import stats
def get_Mode_Array_EqualSize(arr):
    '''
    if np.any(arr)==6:
        return 6
    elif np.any(arr)==2:
        return 2
    elif np.any(arr)==3:
        return 3
    elif np.any(arr)==4:
        return 4
    else:'''
    unique, counts = np.unique(arr, return_counts=True)
    index=np.where(counts==max(counts))
    return unique[index[0]][0]
    #if unique[index[0]][0]==4:return 5
    #else: return 0

def generate_dataset(qsize=32,glb_img_name ="120A",glb_file_path="F:/ImgLib/",repeat=4,cut=False,gridsize=32):
    psize=352
    newdir(glb_file_path+glb_img_name+'/Class_Input')#ok
    newdir(glb_file_path+glb_img_name+'/Class_Output')#ok
    #read data
    print("Pre-Processing,Loading input and output image pair:",glb_img_name+'@halfH.jpg and ',glb_img_name+'@H.jpg')
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
    Class_image[Class_image!=4]=0
    Class_image[Class_image==4]=255

     # if 4 rotation 0 90 180 270
    num=int((2*imageH/qsize-1)*(2*imageW/psize-1))*repeat
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
def U_net(x_train,y_train,qsize,n_epochs,batch_size,X_test,Y_test,early_stop=5,valid_split=0.5):#add ealry_stop 2019.09.07 #50 accuracny
    psize=352
    x_train=(x_train/255.)
    y_train=(y_train/255.)
    input_shape=Input((None, None, 3))

    print('Patch[,],Dataset,epochs,batch,valid,early',qsize,psize,x_train.shape,n_epochs,batch_size,valid_split,early_stop)

    conv1=Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(input_shape)
    conv1=Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)

    conv2=Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    conv2=Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)

    conv3=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    conv3=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv3)
    pool3=MaxPooling2D(pool_size=(2,2))(conv3)

    conv4=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
    conv4=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv4)
    drop4=Dropout(0.5)(conv4)
    pool4=MaxPooling2D(pool_size=(2,2))(drop4)

    conv5=Conv2D(1024,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool4)
    conv5=Conv2D(1024,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv5)
    drop5=Dropout(0.5)(conv5)

    up6=Conv2D(512,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop5))
    merge6=concatenate([drop4,up6],axis=3)
    conv6=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge6)
    conv6=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv6)

    up7=Conv2D(256,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
    merge7=concatenate([conv3,up7],axis=3)
    conv7=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge7)
    conv7=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv7)

    up8=Conv2D(128,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    merge8=concatenate([conv2,up8],axis=3)
    conv8=Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge8)
    conv8=Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv8)

    up9=Conv2D(64,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
    merge9=concatenate([conv1,up9],axis=3)
    conv9=Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge9)
    conv9=Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv9)
    conv9=Conv2D(2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv9)
    conv10=Conv2D(1,1,activation='sigmoid')(conv9)

    model=Model(inputs=input_shape,outputs=conv10)

    opt=Adam(1e-4)
    model.summary()

    model=multi_gpu_model(model)
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=[IOU_calc,'accuracy'])

    callbacks=[EarlyStopping(monitor='val_loss',patience=early_stop)]# 0 no 1 bar 2 line
    history=model.fit(x_train,y_train,batch_size=batch_size,epochs=n_epochs,verbose=1,shuffle=True,validation_split=valid_split,callbacks=callbacks)

    del x_train,y_train
    gc.collect()
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
    plt.savefig(glb_file_path+'_U_Net_'+str(qsize)+'_loss.svg')
    print("loss_history saved")
    plt.close(fig='all')
    #plt.show()#//8/13
    data=pd.DataFrame(history.history)  #为了能够使这组数据成为可以让pandas处理的数据，需要通过这个数组创建DataFrame。
    data.to_csv(glb_file_path+'_U_Net_'+str(qsize)+'_loss.csv',index=True,header=True)

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
    plt.savefig(glb_file_path+'_U_Net_'+str(qsize)+'_acc.svg')
    print("acc_history saved")
    plt.close(fig='all')
    #plt.show()#//8/13
    data=pd.DataFrame(history.history)  #为了能够使这组数据成为可以让pandas处理的数据，需要通过这个数组创建DataFrame。
    data.to_csv(glb_file_path+'_U_Net_'+str(qsize)+'_acc.csv',index=True,header=True)

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
    plt.savefig(glb_file_path+'_U_Net_'+str(qsize)+'_IoU.svg')
    print("IoU_history saved")
    plt.close(fig='all')
    #plt.show()#//8/13
    data=pd.DataFrame(history.history)  #为了能够使这组数据成为可以让pandas处理的数据，需要通过这个数组创建DataFrame。
    data.to_csv(glb_file_path+'_U_Net_'+str(qsize)+'_IoU.csv',index=True,header=True)


def Predication(deepele,qsize=32,glb_img_name ="120A",glb_file_path="F:/ImgLib/",fullH=False,cut=False,resize=True):
    psize=352
    newdir(glb_file_path+glb_img_name+'/Class_Input')
    newdir(glb_file_path+glb_img_name+'/Class_Output')
    bool_h_half=True
    if fullH:
        inputname="H.jpg"
    else:
        inputname="halfH.jpg"
    #read data
    # 载入input图像@EHENCED_Class_image
    if os.path.exists(glb_file_path+glb_img_name+inputname):
        Ortho_image = cv.imread(glb_file_path+glb_img_name+inputname,1)# 导入图片1/2H
        if resize: Ortho_image = cv.resize(Ortho_image,(int(Ortho_image.shape[1]/2),int(Ortho_image.shape[0]/2)),interpolation = cv.INTER_AREA)#@04.21.2019
        imageH,imageW=Ortho_image.shape[0],Ortho_image.shape[1]  #
        imageH=qsize*int(imageH/qsize)
        imageW=psize*int(imageW/psize)
        Ortho_image=Ortho_image[0:imageH,0:imageW]
        cv.imwrite(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+str(qsize)+'_'+str(resize)+'_Class_ReSize_Ortho_image_'+inputname,Ortho_image,[int(cv.IMWRITE_JPEG_QUALITY),100])
        print("Resize and divisable by patch size")
    else:
        Ortho_image=cv.imread(glb_file_path+glb_img_name+'/'+glb_img_name+'Ortho_image.jpg',1)  # 导入图片1/2H
        imageH,imageW=Ortho_image.shape[0],Ortho_image.shape[1]  #
        if cut: Ortho_image=Ortho_image[16:imageH-16,16:imageW-16] # mustcut
        bool_h_half=False
        print("File is not existing")

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

    imageH,imageW=Ortho_image.shape[0],Ortho_image.shape[1]#

    color_o_image=cv.cvtColor(Ortho_image,cv.COLOR_BGR2RGB)
    #Ortho_image= cv.cvtColor(Ortho_image,cv.COLOR_BGR2GRAY)#转换为恢复图片
    print("Ortho-image shape",Ortho_image.shape)

    num=int((2*imageH/qsize-1)*(2*imageW/psize-1))    #print("num",num)
    X=np.zeros(shape=(num,qsize,psize,3))
    A=np.zeros(shape=(num,qsize,psize,3))
    C=[]#reocord ortho-image partial

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
                i=i+1
                if overlap:n=n+int(psize/2)
                else: n=n+psize
            if overlap:m=m+int(qsize/2)
            else: m=m+qsize

    pred_Patch_Class=deepele.predict(A/255.)# predication result
    pred_overlap=deepele.predict(X/255.)
    pred_Assembly=Ortho_image[:,:,0].copy()
    pred_Assembly[:]=0#拼接结果

    i=0
    m=0
    while m<=imageH-qsize:
        n=0
        while n<=imageW-psize:  #m row; n col
            patchClass=np.zeros((qsize,psize))
            patchClass[:]=int(np.argmax(pred_Patch_Class[i]))
            pred_Assembly[m:m+qsize,n:n+psize]=patchClass
            i=i+1
            n=n+psize
        m=m+qsize
    print('sidebyside_Full From Pics',i)
    SideBySide=pred_Assembly.copy()
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
    if bool_h_half:
        cv.imwrite(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+str(qsize)+'_'+str(resize)+'_ReSize_Pred_Class_'+inputname,pred_Assembly,[int(cv.IMWRITE_JPEG_QUALITY),100])
        #save label image 2019.09.02
        data=pd.DataFrame(pred_Assembly)  #pandas DataFrame。
        data.to_csv(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+str(qsize)+'_'+str(resize)+'_ReSize_Pred_Class.csv',index=False,header=False)  #2019.09.02
    else:
        cv.imwrite(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+str(qsize)+'_'+str(resize)+'_ReSize_Pred_Class.jpg',pred_Assembly,[int(cv.IMWRITE_JPEG_QUALITY),100])
        #save label image 2019.09.02
        data=pd.DataFrame(pred_Assembly)  #pandas DataFrame。
        data.to_csv(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+str(qsize)+'_'+str(resize)+'_ReSize_Pred_Class.csv',index=False,header=False)  #2019.09.02
    kl=2
    i=0
    n=5
    plt.figure(figsize=(15,5),dpi=600)
    for i in range(n):
        # plot original
        ax=plt.subplot(2,n,i+1)
        plt.imshow(C[i+kl])
        ax=plt.subplot(2,n,i+1+n)
        ptC=np.zeros((qsize,psize))
        ptC[:]=int(np.argmax(pred_Patch_Class[i+kl]))# The biggest from [0,6]
        plt.imshow(ptC, vmin=-0.5,vmax=9.5,cmap='tab10')
    plt.tight_layout()
    #plt.show()
    plt.savefig(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+'Class Model Predication_'+str(qsize)+'_'+str(resize)+'_Sample.svg')

    plt.figure(figsize=(20,10),dpi=600)
    ax1=plt.subplot(1,3,1)
    im1=plt.imshow(SideBySide,vmin=-0.5,vmax=9.5,cmap='tab10')
    ax1.set_xlabel("Side-by-Side")
    ax2=plt.subplot(1,3,2)
    im2=plt.imshow(pred_Assembly,vmin=-0.5,vmax=9.5,cmap='tab10')
    ax2.set_xlabel("Overlapping")
    ax3=plt.subplot(1,3,3)
    plt.imshow(color_o_image)
    ax3.set_xlabel("Ortho_image")
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
    plt.tight_layout()
    #plt.show()
    plt.savefig(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+'Class Model Predication_'+str(qsize)+'_'+str(resize)+'_Comparing.svg')
    plt.close(fig='all')
    del X,A,C

def Evaluation(deepele,qsize=32,glb_img_name ="120A",glb_file_path="F:/ImgLib/"):
    psize=352
    newdir(glb_file_path+glb_img_name+'/Class_Input')
    newdir(glb_file_path+glb_img_name+'/Class_Output')
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

    imageH,imageW=Ortho_image.shape[0],Ortho_image.shape[1]#

    color_o_image=cv.cvtColor(Ortho_image,cv.COLOR_BGR2RGB)
    #Ortho_image= cv.cvtColor(Ortho_image,cv.COLOR_BGR2GRAY)#转换为恢复图片


    num=int((2*imageH/qsize-1)*(2*imageW/psize-1))    #print("num",num)
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

    pred_Patch_Class=deepele.predict(A/255.) # prediction result
    pred_overlap=deepele.predict(X/255.)
    pred_Assembly=Ortho_image[:,:,0].copy()
    pred_Assembly[:]=0#拼接结果
    i=0
    m=0
    while m<=imageH-qsize:
        n=0
        while n<=imageW-psize:  #m row; n col
            pred_Assembly[m:m+qsize,n:n+psize]=pred_Patch_Class[i].reshape(qsize,psize)*255
            i=i+1
            n=n+psize
        m=m+qsize
    print('sidebyside_Full From Pics',i)
    SideBySide=pred_Assembly.copy()# save as sidebyside result
    array_element_counters(SideBySide)
    SideBySide[SideBySide<255/2]=0
    SideBySide[SideBySide>=255/2]=255
    array_element_counters(SideBySide)
    #update center part; overlaping
    i=0
    m=0
    while m<=imageH-qsize:
        n=0
        while n<=imageW-psize:  #m row; n col
            overlap_patchClass=np.zeros((qsize,psize))
            overlap_patchClass[:]=pred_overlap[i].reshape(qsize,psize)*255
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
    pred_Assembly[pred_Assembly<255/2]=0
    pred_Assembly[pred_Assembly>=255/2]=255
    array_element_counters(pred_Assembly)

    cv.imwrite(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+str(qsize)+'_U_Net_.jpg',pred_Assembly,[int(cv.IMWRITE_JPEG_QUALITY),100])
    #save label image 2019.09.02
    data=pd.DataFrame(pred_Assembly)  #pandas DataFrame。
    data.to_csv(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+str(qsize)+'_U_Net_.csv',index=False,header=False)  #2019.09.02

    kl=0
    i=0
    n=8
    plt.figure(figsize=(15,5),dpi=600)
    for i in range(n):
        # plot original
        ax=plt.subplot(4,n,i+1)
        plt.imshow(C[i+kl])
        ax=plt.subplot(4,n,i+1+n*1)
        plt.imshow(E[i+kl],vmin=-0.5,vmax=9.5,cmap='tab10')
        ax=plt.subplot(4,n,i+1+n*2)
        ptC=np.zeros((qsize,psize))
        ptC[:]=pred_Patch_Class[i+kl].reshape(qsize,psize)*255
        plt.imshow(ptC,cmap='gray')
        ax=plt.subplot(4,n,i+1+n*3)
        ptC_b=ptC.copy()
        ptC_b[ptC_b<255/2]=0
        ptC_b[ptC_b>=255/2]=255
        plt.imshow(ptC_b,cmap='gray')
    plt.tight_layout()
    #plt.show()
    plt.savefig(glb_file_path+glb_img_name+'/'+glb_img_name+'_U_Net_'+str(qsize)+'_Sample.svg')

    plt.figure(figsize=(15,15),dpi=600)
    ax1=plt.subplot(2,2,1)
    im1=plt.imshow(SideBySide,cmap='gray')
    ax1.set_xlabel("Side-by-Side")
    ax2=plt.subplot(2,2,2)
    im2=plt.imshow(pred_Assembly,cmap='gray')
    ax2.set_xlabel("Overlapping")
    ax3=plt.subplot(2,2,3)
    im3=plt.imshow(Class_image,vmin=-0.5,vmax=9.5,cmap='tab10')
    ax3.set_xlabel("Label_image")
    ax4=plt.subplot(2,2,4)
    plt.imshow(color_o_image)
    ax4.set_xlabel("Ortho_image")

    # Colorbar
    '''
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
    '''
    divider3=make_axes_locatable(ax3)#ax3
    cax3=divider3.append_axes("right",size="1.5%",pad=0.05)  #colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
    cbar3=plt.colorbar(im3,cax=cax3,cmap='tab10')
    cbar3.set_ticks(colorlist)  # color bar
    cbar3.ax.get_yaxis().set_ticks([])
    for j,lab in enumerate(colorLabel):
        cbar3.ax.text(10,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center')  #                         colorlist[0],colorLabel[0])
    plt.tight_layout()
    #plt.show()
    plt.savefig(glb_file_path+glb_img_name+'/'+glb_img_name+'_U_Net_'+str(qsize)+'_Comparing.svg')
    del X,A,C,E
    plt.close(fig='all')

def EvaluationLargeSize(deepele,qsize=32,glb_img_name ="120A",glb_file_path="F:/ImgLib/"):
    psize=352
    newdir(glb_file_path+glb_img_name+'/Class_Input')
    newdir(glb_file_path+glb_img_name+'/Class_Output')
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

    imageH,imageW=Ortho_image.shape[0],Ortho_image.shape[1]#

    color_o_image=cv.cvtColor(Ortho_image,cv.COLOR_BGR2RGB)
    #Ortho_image= cv.cvtColor(Ortho_image,cv.COLOR_BGR2GRAY)#转换为恢复图片

    C=[]#reocord ortho-image partial
    E=[]#reocord Elevation-map partial


    A=Ortho_image.reshape((1,imageH,imageW,3))
    C.append(color_o_image)
    E.append(Class_image)

    pred_Assembly=deepele.predict(A/255.)*255 # prediction result
    pred_Assembly=pred_Assembly.reshape((imageH,imageW,1))
    #array_element_counters(pred_Assembly)
    pred_Assembly[pred_Assembly<255/4]=0
    pred_Assembly[pred_Assembly>=255/4]=255
    array_element_counters(pred_Assembly)

    cv.imwrite(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+str(qsize)+'_U_Net_Large.jpg',pred_Assembly,[int(cv.IMWRITE_JPEG_QUALITY),100])
    #save label image 2019.09.02
    data=pd.DataFrame(pred_Assembly)  #pandas DataFrame。
    data.to_csv(glb_file_path+glb_img_name+'/'+glb_img_name+'_'+str(qsize)+'_U_Net_Large.csv',index=False,header=False)  #2019.09.02

def repeatmodeloutput(qsize,n_epochs,batch_size,glb_file_path,early_stop):
    RepNum=4
    psize=352
    GSize=min(qsize,psize)//2

    Inputlist=['0','1','2','3','4','6']#,'12','13','14']
    X_train=[]#None,None,None,None,None,None,None,None,None]
    Y_train=[]#None,None,None,None,None,None,None,None,None]
    C_train=[]
    for i in range(len(Inputlist)):
        X_input_c,Y_input_c, C_input_c =generate_dataset(qsize,glb_img_name=Inputlist[i],glb_file_path=glb_file_path,repeat=RepNum,gridsize=GSize)[0:3]
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

    deepelev1,historyplot =U_net(X_train,Y_train,qsize,n_epochs,batch_size,X_t,C_t,early_stop=early_stop) # neetwork self normalazion ClassficationV1(X_train,Y_train,qsize,n_epochs,batch_size,X_test,Y_test)

    del X_train,Y_train,C_train
    gc.collect()

    printloss(historyplot,glb_file_path=glb_file_path,qsize=qsize)
    printacc(historyplot,glb_file_path=glb_file_path,qsize=qsize)
    printIoU(historyplot,glb_file_path=glb_file_path,qsize=qsize)
    print("=====Patch:",qsize,"============================================================================================================Evaluation")
    plt.close(fig='all')
    for i in range(len(Inputlist)):
        Evaluation(deepelev1,qsize,glb_img_name=Inputlist[i],glb_file_path=glb_file_path)
        EvaluationLargeSize(deepelev1,qsize,glb_img_name=Inputlist[i],glb_file_path=glb_file_path)
    plt.close(fig='all')
    print("=====Patch:",qsize,"============================================================================================================Testing")
    EvaluationLargeSize(deepelev1,qsize,glb_img_name='7',glb_file_path=glb_file_path)
    EvaluationLargeSize(deepelev1,qsize,glb_img_name='8',glb_file_path=glb_file_path)
    EvaluationLargeSize(deepelev1,qsize,glb_img_name='9',glb_file_path=glb_file_path)
    EvaluationLargeSize(deepelev1,qsize,glb_img_name='11',glb_file_path=glb_file_path)
    EvaluationLargeSize(deepelev1,qsize,glb_img_name='12',glb_file_path=glb_file_path)
    EvaluationLargeSize(deepelev1,qsize,glb_img_name='13',glb_file_path=glb_file_path)
    EvaluationLargeSize(deepelev1,qsize,glb_img_name='14',glb_file_path=glb_file_path)
    EvaluationLargeSize(deepelev1,qsize,glb_img_name='E1',glb_file_path=glb_file_path)
    EvaluationLargeSize(deepelev1,qsize,glb_img_name='E2',glb_file_path=glb_file_path)
    EvaluationLargeSize(deepelev1,qsize,glb_img_name='E3',glb_file_path=glb_file_path)

    '''    
    print("=====Patch:",qsize,"============================================================================================================Predication_ReSize")
    Testlist=['20AO','40AO','80AO']
    for i in range(len(Testlist)):
        Predication(deepelev1,qsize,glb_img_name=Testlist[i],glb_file_path=glb_file_path,fullH=False,resize=True)#,resize=False
    plt.close(fig='all')
    print("=====Patch:",qsize,"============================================================================================================Prediction_Non-resize")
    for i in range(len(Testlist)):
        Predication(deepelev1,qsize,glb_img_name=Testlist[i],glb_file_path=glb_file_path,fullH=True,resize=False)#,resize=False
    '''
    del deepelev1
    gc.collect()

#=============================================================================
if __name__ == '__main__':
    #Running Parmaters   1212
    #-----------------------------
    if os.path.exists('D:/'):
        glb_file_path='D:/CentOS/GoogleEarthHighway/'
        b='win'
        cpu=3
    elif os.path.exists('/data/'):
        glb_file_path='/data/GoogleEarthHighway/'
        b='server'
        cpu=4
    print(glb_file_path,end=' ')
    #glb_img_name='20A'
    qsize=512# best 32
    n_epochs=100  # 1000
    early_stop=10

    batch_size=16
    plt.close(fig='all')
    print("=====Patch:",qsize,"=====================================================================")
    repeatmodeloutput(qsize,n_epochs,batch_size,glb_file_path,early_stop)
    #qsize=qsize//2
    gc.collect()

    #import Show_Profile_GUI_halfH