#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
# Copyright:    Yuhan Jiang
# Email:        yuhan.jiang@marquette.edu
# Date:         10/03/2020
# major update : add GPS information
# next update:
#
from datetime import datetime
start_time=datetime.now()
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import sys
import tkinter as tk
import csv,re
import pandas as pd
from tkinter import filedialog
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.ndimage import filters
def get_filePath_fileName_fileExt(filename):
    (filepath,tempfilename) = os.path.split(filename);
    (shotname,extension) = os.path.splitext(tempfilename);
    return filepath,shotname,extension
def array_element_counters(a):
    unique, counts = np.unique(a, return_counts=True)
    counters=dict(zip(unique, counts))
    print('max,min,num#',max(unique),min(unique),len(unique))
    print('Dic:',counters)
    return counters
def array_element_mode(a):
    unique, counts = np.unique(a, return_counts=True)
    index=np.where(counts==max(counts))
    return unique[index[0]]

def add_Py3D_log(glb_file_path,str_data):
    path_file_name = glb_file_path+'HighwayASS_log.txt'
    if not os.path.exists(path_file_name):
        with open(path_file_name, "w") as f:
            print(f)
    with open(path_file_name, "a") as f:
        f.writelines(str_data)

def Repeat(glb_img_name,Project_bool=True,GPS_bool=False,GPSinf=None):

    if Project_bool:
        glb_file_path=glb_project_path+glb_img_name
        public_key_path=glb_project_path

    else:
        glb_file_path=glb_training_path+glb_img_name
        public_key_path=glb_training_path

    Ortho_image = cv.imread(glb_file_path+"/"+glb_img_name+"Ortho_image.jpg",1)# 导入图片1/2H

    filename=str(glb_file_path+"/"+glb_img_name+"Ortho_image.jpg")
    Ortho_image=cv.cvtColor(Ortho_image,cv.COLOR_BGR2RGB)
    imageH,imageW=Ortho_image.shape[0:2]
    Label_image=np.zeros((imageH,imageW))
    Label_Img=cv.cvtColor(Ortho_image,cv.COLOR_RGB2GRAY)

    if Project_bool==False:
        # read label image 2019.09.02
        PixelP=pd.read_csv(glb_file_path+"/"+glb_img_name+'Label_image.csv',index_col=False,header=None)
        print('number of Pixels Loaded:',len(PixelP),len(PixelP[1]))
        Label=np.array(PixelP).reshape((imageH,imageW)).copy()  # 2019.09.02
        Label_image=np.zeros((imageH,imageW))
        Label_image[:]=Label[:imageH,:imageW]
        print('Label_image Successful! shape of Label_image:',Label_image.shape)
        Label_image[Label_image!=4]=0
        Label_image[Label_image==4]=1
    else:
        # read label image prediction 2020.05.11
        try:
            PixelP=pd.read_csv(glb_file_path+"/"+glb_img_name+'_'+str(Split)+'_Evaluation_Pred_Class.csv',index_col=False,header=None)
            print('number of Pixels Loaded:',len(PixelP),len(PixelP[1]))
            Label=np.array(PixelP).reshape((imageH,imageW)).copy()  # 2019.09.02
            Label_image=np.zeros((imageH,imageW))
            Label_image[:]=Label[:imageH,:imageW]
            print('Label_image Successful! shape of Label_image_Prediction:',Label_image.shape)
            Label_image[Label_image!=4]=0
            Label_image[Label_image==4]=1
        except:
            print('Not DCNN Prediction')
            PixelP=pd.read_csv(glb_file_path+"/"+glb_img_name+'_'+str(Split)+'_U_Net_.csv',index_col=False,header=None)
            print('number of Pixels Loaded:',len(PixelP),len(PixelP[1]))
            Label=np.array(PixelP).reshape((imageH,imageW)).copy()  # 2019.09.02
            Label_image=np.zeros((imageH,imageW))
            Label_image[:]=Label[:imageH,:imageW]
            print('Label_image Successful! shape of U-Net_Prediction:',Label_image.shape)
            Label_image[Label_image!=255]=0
            Label_image[Label_image==255]=1
            Label[Label!=255]=0
            Label[Label==255]=4
    if True:
        Label_Img[:]=Label_image
        #Label_image=cv.Canny(Label_Img,0,1)
        Longitudinal_Cracking=np.zeros((imageH,imageW))
        Transverse_Cracking=np.zeros((imageH,imageW))

        filters.sobel(Label_Img,1,Longitudinal_Cracking)
        filters.sobel(Label_Img,0,Transverse_Cracking)

        Longitudinal_Cracking[Longitudinal_Cracking!=-4]=0
        Transverse_Cracking[Transverse_Cracking!=-4]=0

        try:
            Longitudinal_Cracking_pixel_number=array_element_counters(Longitudinal_Cracking)[-4.0]# two pixel one line
        except:
            Longitudinal_Cracking_pixel_number=0
        try:
            Transverse_Cracking_pixel_number=array_element_counters(Transverse_Cracking)[-4.0] # two pixel one line
        except:
            Transverse_Cracking_pixel_number=0
        CrackW=15#width=14.
        line_pixel=100-CrackW # fix the hole
        T_C_number=Transverse_Cracking_pixel_number/2/ line_pixel
        Longitudinal_Crack_in_ft=(Longitudinal_Cracking_pixel_number/2+T_C_number*CrackW)*(139/1048)
        Longitudinal_Cracking_index=Longitudinal_Crack_in_ft/(1024/1048*139)*100/2# line ft/100ft single direction(/2
        Transverse_Cracking_index=T_C_number/(1024/1048*139) *100/2 # count /100ft single direction (/2

        Label_image[:]=Label[:imageH,:imageW]
        print('Labelimage,Longitudinal_Cracking_index,Transverse_Cracking_index,Transverse_Cracking_Number',Longitudinal_Cracking_index,Transverse_Cracking_index,T_C_number)

        if GPS_bool==False or GPSinf.all()==None:
            add_Py3D_log(public_key_path,[glb_img_name,'\t',str(psize),'\t',str(imageH),'\t',str(format(Longitudinal_Cracking_index,'.2f')),'\t',str(format(Transverse_Cracking_index,'.2f')),'\t',str(format(T_C_number,'.1f')),'\t'
                ,start_time.__str__(),'\t', datetime.now().__str__(),'\t',str(filename),'\n'])
        else:
            CameraHeight=GPSinf[int(glb_img_name)][1]
            Latitude_float=GPSinf[int(glb_img_name)][2]
            Longitude_float=GPSinf[int(glb_img_name)][3]

            add_Py3D_log(public_key_path,[glb_img_name,'\t',str(psize),'\t',str(imageH),'\t',str(format(Longitudinal_Cracking_index,'.2f')),'\t',str(format(Transverse_Cracking_index,'.2f')),'\t',str(format(T_C_number,'.1f')),'\t'
                ,start_time.__str__(),'\t',datetime.now().__str__(),'\t',str(filename),'\t',str(CameraHeight),'\t',str(Latitude_float),'\t',str(Longitude_float),'\n'])

    print('Label_image Freq.')
    array_element_counters(Label_image)
    lastsetp_Label_image=Label_image.copy()

    if os.path.exists(public_key_path+'Public_Object_Label_Dic.csv'):
        print(public_key_path+'Public_Object_Label_Dic.csv')
        colordic={}
        with open(public_key_path+'Public_Object_Label_Dic.csv') as f:
            colordic=dict(filter(None,csv.reader(f)))
            for key,val in colordic.items():
                colordic[key]=int(val)
        print('Public Label Dic Loaded',colordic)
        #print(colordic)
    else:
        colordic={'Default':0, 'Pavement/Line Mark/Bridge/OtherPavementSurface':1,'Truck/Bus/Car':2,'Light/TrafficSign':3,'Crack':4,'VegetationZoo':5}
        print(colordic)
        colordic={'d':0, 'p':1,'t':2,'l':3,'c':4,'v':5}
        print(colordic)
        print("Created New Object Label Dic")

    imageH,imageW=Ortho_image.shape[0:2]

    fig=plt.figure("HighwayASS_Project"+glb_img_name,figsize=(18,8))
    #plt.ion()
    ax1=plt.subplot(2,2,1)
    ax1.imshow(Ortho_image)
    ax1.set_title("Google Earth Imagery")
    ax2=plt.subplot(2,2,2)
    im=ax2.imshow(Label_image,vmin=-0.5,vmax=9.5,cmap='tab10')
    ax2.set_title("Label_image")
    ax3=plt.subplot(2,2,3)
    ax3.imshow(Longitudinal_Cracking,cmap='gray')
    ax3.set_title("Longitudinal Cracks")
    ax3.set_xlabel("Longitudinal Cracking:"+str(format(Longitudinal_Cracking_index,'.2f'))+' ft/100-ft. station')#print(format((10/3),'.1f'))
    ax4=plt.subplot(2,2,4)
    ax4.imshow(Transverse_Cracking,cmap='gray')
    ax4.set_title("Transverse Cracks")
    ax4.set_xlabel('#'+str(format(T_C_number,'.1f'))+" Transverse Cracking:"+str(format(Transverse_Cracking_index,'.2f'))+' /100-ft. station')

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider=make_axes_locatable(ax2)
    cax=divider.append_axes("right",size="3%",pad=0.05)#colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
    cbar= plt.colorbar(im,cax=cax,cmap='terrain')
    colorlist=list(map(int,list(colordic.values())))
    colorLabel=list(map(str,list(colordic.keys())))
    cbar.set_ticks(colorlist)  # color bar
    cbar.ax.get_yaxis().set_ticks([])
    for j,lab in enumerate(colorLabel):
        cbar.ax.text(2.5,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center',color='b')  #                         colorlist[0],colorLabel[0])

    plt.tight_layout()
    plt.savefig(glb_file_path+'/'+glb_img_name+'CrackingRating.svg')
    #plt.pause(1.5)  # show image for 1.5 sec plt.show()            time.sleep(2)
    #plt.ioff()  #turn off interaction mode, avoid
    plt.clf()  #clean image
    plt.close(fig)  #close window

    #check=pd.read_csv(glb_file_path+"/"+glb_img_name+'Label_image.csv',index_col=False,header=None)
    #print('number of Pixels Loaded:',len(check),len(check[1]))
    #check_image=np.array(check).reshape((imageH,imageW)).copy()  # 2019.09.02
    #array_element_counters(check_image)
###################add
if __name__ == '__main__':

    TrainingList=[0,1,2,3,4,6]
    ProjectList=list(range(53))  # 37 is the number of image in the project file for eveluation
    Split=0.5
    psize=16
    if os.path.exists('D:/'):
        glb_training_path='D:/CentOS/G_Training/'
        glb_project_path='D:/CentOS/G2/'
        b='win'
        cpu=3
    elif os.path.exists('/data/'):
        glb_training_path='/data/G_Training/'
        glb_project_path='/data/G3/'
        b='server'
        cpu=4

    GpsPD=pd.read_csv(glb_project_path+'HighwayASS_GPS.txt',sep='\t',header=0)#read GPS information for the project
    GpsInf=np.array(GpsPD)#    print(GpsInf)

    '''
    print(glb_training_path,end=' ')
    print("Traning Image")
    add_Py3D_log(glb_project_path,['ID','\t','PatchSize','\t','imageHeight','\t','Longitudinal_Cracking_index','\t','Transverse_Cracking_index','\t','T_C_number','\t','Start_time','\t','End_time','\t','File_name','\t','Camera','\t','Latitude_float','\t','Longitude_float','\n'])

    for i in TrainingList:
        Repeat(str(i),Project_bool=False)
    '''

    print(glb_project_path,end=' ')
    print("Project Image")
    add_Py3D_log(glb_project_path,['ID','\t','PatchSize','\t','imageHeight','\t','Longitudinal_Cracking_index','\t','Transverse_Cracking_index','\t','T_C_number','\t','Start_time','\t','End_time','\t','File_name','\t','Camera','\t','Latitude_float','\t','Longitude_float','\n'])
    for i in ProjectList:
        Repeat(str(i),Project_bool=True,GPS_bool=True,GPSinf=GpsInf)