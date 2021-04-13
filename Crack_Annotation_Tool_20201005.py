#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
# Copyright:    Yuhan Jiang
# Email:        yuhan.jiang@marquette.edu
# Date:         10/05/2020
# major update :
#
from datetime import datetime
start_time=datetime.now()
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import filters

def Repeat(glb_img_name,prediction=0,Annotate_bool=True):
    if prediction==16 or prediction==8:
        glb_file_path="D:/CentOS/GoogleEarthHighway_CNN/"+glb_img_name
    else:
        glb_file_path="D:/CentOS/GoogleEarthHighway_U_net_crack255_early/"+glb_img_name

    if os.path.exists('D:/'):
        public_key_path='D:/CentOS/GoogleEarthHighway/'
    elif os.path.exists('/data/'):
        public_key_path='/data/GoogleEarthHighway/'

    Ortho_image = cv.imread(glb_file_path+"/"+glb_img_name+"Ortho_image.jpg",1)# 导入图片1/2H

    filename=str(glb_file_path+"/"+glb_img_name+"Ortho_image.jpg")
    Ortho_image=cv.cvtColor(Ortho_image,cv.COLOR_BGR2RGB)
    imageH,imageW=Ortho_image.shape[0:2]

    Label_Img=cv.cvtColor(Ortho_image,cv.COLOR_RGB2GRAY)

    if prediction==0:
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
            PixelP=pd.read_csv(glb_file_path+"/"+glb_img_name+'_'+str(prediction)+'_Evaluation_Pred_Class.csv',index_col=False,header=None)
            print('number of Pixels Loaded:',len(PixelP),len(PixelP[1]))
            Label=np.array(PixelP).reshape((imageH,imageW)).copy()  # 2019.09.02
            Label_image=np.zeros((imageH,imageW))
            Label_image[:]=Label[:imageH,:imageW]
            print('Label_image Successful! shape of Label_image_Prediction:',Label_image.shape)
            Label_image[Label_image!=4]=0
            Label_image[Label_image==4]=1
        except:
            PixelP=pd.read_csv(glb_file_path+"/"+glb_img_name+'_'+str(prediction)+'_U_Net_.csv',index_col=False,header=None)
            print('number of Pixels Loaded:',len(PixelP),len(PixelP[1]))
            Label=np.array(PixelP).reshape((imageH,imageW)).copy()  # 2019.09.02
            Label_image=np.zeros((imageH,imageW))
            Label_image[:]=Label[:imageH,:imageW]
            print('Label_image Successful! shape of U-Net_Prediction:',Label_image.shape)
            Label_image[Label_image!=255]=0
            Label_image[Label_image==255]=1
            Label[Label!=255]=0
            Label[Label==255]=4


    if os.path.exists(public_key_path+'Public_Object_Label_Dic.csv'):
        print(public_key_path+'Public_Object_Label_Dic.csv')
        with open(public_key_path+'Public_Object_Label_Dic.csv') as f:
            colordic=dict(filter(None,csv.reader(f)))
            for key,val in colordic.items():
                colordic[key]=int(val)
        print('Public Label Dic Loaded',colordic)
    else:
        colordic={'Default':0, 'Pavement/Line Mark/Bridge/OtherPavementSurface':1,'Truck/Bus/Car':2,'Light/TrafficSign':3,'Crack':4,'VegetationZoo':5}
        print(colordic)
        colordic={'d':0, 'p':1,'t':2,'l':3,'c':4,'v':5}
        print(colordic)
        print("Created New Object Label Dic")

    if Annotate_bool==True:
        Label_Img[:]=Label_image
        Longitudinal_Cracking=np.zeros((imageH,imageW))
        Transverse_Cracking=np.zeros((imageH,imageW))

        filters.sobel(Label_Img,1,Longitudinal_Cracking)
        filters.sobel(Label_Img,0,Transverse_Cracking)

        #Longitudinal_Cracking[Longitudinal_Cracking!=-4]=0
        #Transverse_Cracking[Transverse_Cracking!=-4]=0

        CrackW=15  #width=14.

        annotated=Ortho_image.copy()
        annotated[Label==4]=(125,255,125)  # annotate cracks
        #for i in [4,3]:
        #   annotated[Longitudinal_Cracking==-i]=(255,0,0)
        #    annotated[Transverse_Cracking==-i]=(0,0,255)
        #    annotated[Longitudinal_Cracking==i]=(255,0,0)
        #    annotated[Transverse_Cracking==i]=(0,0,255)

        bb=2

        for u in range(imageW):
            for v in range(imageH):
                if Longitudinal_Cracking[v,u] in [4,3]: # annotate longitudianl cracks left edges
                    annotated[v,u:u+bb]=(255,0,0)
                if Longitudinal_Cracking[v,u] in [-3,-4]: # annotate longitudianl cracks right edges
                    annotated[v,u-bb:u]=(255,0,0)
                if Transverse_Cracking[v,u] in [4,3]:# annotate transverse cracks upper edges
                    annotated[v:v+bb,u]=(0,0,255)
                if Transverse_Cracking[v,u] in [-3,-4]: # annotate transverse cracks bottom edges
                    annotated[v-bb:v,u]=(0,0,255)

        fig=plt.figure("HighwayASS_Project"+glb_img_name,figsize=(18,4))
        ax1=plt.subplot(1,3,1)
        ax1.imshow(Ortho_image)
        ax1.set_title("Google Earth Imagery")
        ax2=plt.subplot(1,3,2)
        im=ax2.imshow(Label,vmin=-0.5,vmax=9.5,cmap='tab10')
        ax2.set_title("Label_image")
        ax3=plt.subplot(1,3,3)
        ax3.imshow(annotated)
        ax3.set_title("Annotated Cracks")

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
        plt.show()
        plt.savefig(glb_file_path+"/"+glb_img_name+"Crack Annotation.svg")
        plt.close()

###################add

listA=['0','1','2','3','4','6','8','9','11','12','13']
#listA=['0','1','2','3','4','6','12','13']
for i in listA:
    Repeat(i,prediction=16,Annotate_bool=True)