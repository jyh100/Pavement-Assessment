#!/usr/bin/env python3.7.3
# -*- coding: utf-8 -*-
# Copyright:    Yuhan Jiang
# Email:        yuhan.jiang@marquette.edu
# Date:         5/15/2020
# major update :IOU, intersection over union
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

#########start 获取文件路径、文件名、后缀名############
def get_filePath_fileName_fileExt(filename):
    (filepath,tempfilename) = os.path.split(filename);
    (shotname,extension) = os.path.splitext(tempfilename);
    return filepath,shotname,extension
#########end 获取文件路径、文件名、后缀名############
def array_element_counters(a):
    unique, counts = np.unique(a, return_counts=True)
    counters=dict(zip(unique, counts))
    #print('max,min,num#',max(unique),min(unique),len(unique))
    print('Dic:',counters)
    return counters
def array_element_mode(a):
    unique, counts = np.unique(a, return_counts=True)
    index=np.where(counts==max(counts))
    return unique[index[0]]
def add_Py3D_log(glb_file_path,str_data):
    path_file_name = glb_file_path+'HighwayASS_IoU.txt'
    if not os.path.exists(path_file_name):
        with open(path_file_name, "w") as f:
            print(f)
    with open(path_file_name, "a") as f:
        f.writelines(str_data)

def GUIshowSINGLE(badbool=True,patchS=32):
    root = tk.Tk()
    root.title("Label_image Comparing V_0.01")#Please Open A Ortho image
    root.geometry("300x200")
    theLabel=tk.Label(root,text='Author: Yuhan Jiang')
    theLabel.pack()#用于自动调节组件自身的尺寸
    theLabel=tk.Label(root,text='E-mail: yuhan.jiang@marquette.edu')
    theLabel.pack()#用于自动调节组件自身的尺寸
    theLabel=tk.Label(root,text='Step 1:Please Open A Label-image')
    theLabel.pack()#用于自动调节组件自身的尺寸
    #root.mainloop()
    root.filename=filedialog.askopenfilename(initialdir="/",title="Select file",filetypes=(("jpg files","*Ortho_image.jpg"),("all files","*.*")))
    #print ("Opened",root.filename)
    glb_file_path,file_name=get_filePath_fileName_fileExt(root.filename)[0:2]
    glb_img_name =file_name[:-11]#"10G" # 20b  print(glb_img_name)
    if os.path.exists('D:/'):
        public_key_path='D:/CentOS/GoogleEarthHighway/'
    elif os.path.exists('/data/'):
        public_key_path='/data/GoogleEarthHighway/'

    Ortho_image = cv.imread(glb_file_path+"/"+glb_img_name+"Ortho_image.jpg",1)# 导入图片1/2H
    Ortho_image=cv.cvtColor(Ortho_image,cv.COLOR_BGR2RGB)
    imageH,imageW=Ortho_image.shape[0:2]
    Label_image=np.zeros((imageH,imageW))

    test_file_path="D:/CentOS/GoogleEarthHighway_U_net_crack255_early/"+glb_img_name

    if os.path.exists(test_file_path+"/"+glb_img_name+'_512_U_Net_.csv'):
        PixelP=pd.read_csv(test_file_path+"/"+glb_img_name+'_512_U_Net_.csv',index_col=False,header=None)
        Label_image=np.array(PixelP).reshape((imageH,imageW)).copy()  # 2019.09.02
        Label_image[Label_image==255]=4
    else:
        print("error")

    print('Label_image Freq.',end='')
    ldic=array_element_counters(Label_image)

    colordic={'d':0,'c':4}

    root.destroy()
    try:
        PixelP=pd.read_csv(glb_file_path+"/"+glb_img_name+'_'+str(patchS)+'_Evaluation_Pred_Class.csv',index_col=False,header=None)
        print('number of Pixels Loaded:',len(PixelP),len(PixelP[1]))
        Label_output=np.array(PixelP).reshape((imageH,imageW)).copy()  # 2019.09.02
        Label_output[Label_output!=4]=0
    except:
        print('error')

    #print('Label_image_output Successful! shape of Label_image:',Label_output.shape,end='')
    print('Label_image_output Freq.',end='')
    pdic=array_element_counters(Label_output)

    Ortho_image_back=Ortho_image.copy()
    Label_image_back=Label_image.copy()
    Label_output_back=Label_output.copy()

    colorlist=list(map(int,list(colordic.values())))
    colorLabel=list(map(str,list(colordic.keys())))
    dichavlist0=list(map(int,list(ldic.keys())))
    dichavlist1=list(map(int,list(pdic.keys())))
    listAll=list(set(dichavlist0)|set(dichavlist1))
    listAll.sort()
    print(glb_img_name,listAll)

    overall=True
    if overall:
        diff=(Label_image-Label_output)==0
        Ortho_image[diff]=255
        Label_image[diff]=255
        Label_output[diff]=255
        print('Overall\tPixel Accuracy:\t',np.sum(diff)/(imageW*imageH),np.sum(diff),'\tof\t',(imageW*imageH))

        if badbool==True:# and np.sum(diff)/(imageW*imageH)<0.5:
            fig=plt.figure("Comparing the Resutls_"+glb_img_name,figsize=(18,8))
            ax1=plt.subplot(2,2,3)

            ax1.imshow(Ortho_image)
            ax1.set_xlabel("Ortho_image")
            ax2=plt.subplot(2,2,1)
            im2=ax2.imshow(Label_image,vmin=-0.5,vmax=9.5,cmap='tab10')
            ax2.set_xlabel("Label_image")

            ax3=plt.subplot(2,2,2)
            im3=ax3.imshow(Label_output,vmin=-0.5,vmax=9.5,cmap='tab10')
            ax3.set_xlabel("Label_image_Patch-"+str(patchS))

            ax4=plt.subplot(2,2,4)
            im4=ax4.imshow(diff,vmin=0,vmax=1,cmap='gray')
            ax4.set_xlabel("Diff_Patch-"+str(patchS))

            #colordic={'wood':(10,0,0),'veg':(20,0,0),'sand':(30,0,0)}
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider2=make_axes_locatable(ax2)
            cax2=divider2.append_axes("right",size="3%",pad=0.05)#colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
            cbar2= plt.colorbar(im2,cax=cax2,cmap='tab10')

            cbar2.set_ticks(colorlist)  # color bar
            cbar2.ax.get_yaxis().set_ticks([])
            for j,lab in enumerate(colorLabel):
                cbar2.ax.text(2.5,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center',color='hotpink')  #                         colorlist[0],colorLabel[0])
            divider3=make_axes_locatable(ax3)
            cax3=divider3.append_axes("right",size="3%",pad=0.05)#colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
            cbar3= plt.colorbar(im3,cax=cax3,cmap='tab10')
            cbar3.set_ticks(colorlist)  # color bar
            cbar3.ax.get_yaxis().set_ticks([])
            for j,lab in enumerate(colorLabel):
                cbar3.ax.text(2.5,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center',color='hotpink')  #                         colorlist[0],colorLabel[0])

            colorlist2=[0,1]#list(map(int,list(colordic.values())))
            colorLabel2=["Error","Matched"]#list(map(str,list(colordic.keys())))
            divider4=make_axes_locatable(ax4)
            cax4=divider4.append_axes("right",size="3%",pad=0.05)#colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
            cbar4= plt.colorbar(im4,cax=cax4,cmap='gray')
            cbar4.set_ticks(colorlist2)  # color bar
            cbar4.ax.get_yaxis().set_ticks([])
            for j,lab in enumerate(colorLabel2):
                cbar4.ax.text(2.5,colorlist2[j],"-"+str(colorlist2[j])+'- '+lab,ha='left',va='center',color='hotpink')  #                         colorlist[0],colorLabel[0])


            plt.tight_layout()
            plt.show()

    for i in listAll:

        Ortho_image=Ortho_image_back.copy()
        Label_image=Label_image_back.copy()
        Label_output=Label_output_back.copy()
        LT=None
        PT=None

        try:
            LT=(Label_image==i)
        except:
            continue
        try:
            PT=(Label_output==i)
        except:
            continue
        Label_image[~LT]=1000
        Label_output[~PT]=1001
        intersect=(Label_image-Label_output)==0
        blank=Label_image.copy()
        blank[:,:]=0
        blank[LT]=1
        blank[PT]=1
        union=(blank==1)
        Ortho_image[~union]=255
        Label_image[~LT]=255
        Label_output[~PT]=255
        print(i,'\tIoU:\t',np.sum(intersect)/np.sum(union),np.sum(intersect),'\tof\t',np.sum(union),'\tLable:\t',np.sum(LT),'\tOutput:\t',np.sum(PT))

        if True:#(badbool==True and np.sum(intersect)/np.sum(union)<0.5 and np.sum(union)!=0) or  np.sum(intersect)==0:#20204.14
            fig=plt.figure("Comparing the Resutls_"+glb_img_name,figsize=(18,8))
            ax1=plt.subplot(2,2,3)
            ax1.imshow(Ortho_image)
            ax1.set_xlabel("Union-"+str(np.sum(union))+'_'+str(i))

            ax2=plt.subplot(2,2,1)
            im2=ax2.imshow(Label_image,vmin=-0.5,vmax=9.5,cmap='tab10')
            ax2.set_xlabel("Label_image-"+str(np.sum(LT))+'_'+str(i))

            ax3=plt.subplot(2,2,2)
            im3=ax3.imshow(Label_output,vmin=-0.5,vmax=9.5,cmap='tab10')
            ax3.set_xlabel("Label_image_Patch-"+str(patchS)+"-"+str(np.sum(PT))+'_'+str(i))

            Ortho_image=Ortho_image_back.copy()
            Ortho_image[~intersect]=255
            ax4=plt.subplot(2,2,4)
            ax4.imshow(Ortho_image)
            ax4.set_xlabel("Intersection-"+str(np.sum(intersect))+'_'+str(i))

            #colordic={'wood':(10,0,0),'veg':(20,0,0),'sand':(30,0,0)}
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider2=make_axes_locatable(ax2)
            cax2=divider2.append_axes("right",size="3%",pad=0.05)#colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
            cbar2= plt.colorbar(im2,cax=cax2,cmap='tab10')

            cbar2.set_ticks(colorlist)  # color bar
            cbar2.ax.get_yaxis().set_ticks([])
            for j,lab in enumerate(colorLabel):
                cbar2.ax.text(2.5,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center',color='hotpink')  #                         colorlist[0],colorLabel[0])
            divider3=make_axes_locatable(ax3)
            cax3=divider3.append_axes("right",size="3%",pad=0.05)#colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
            cbar3= plt.colorbar(im3,cax=cax3,cmap='tab10')
            cbar3.set_ticks(colorlist)  # color bar
            cbar3.ax.get_yaxis().set_ticks([])
            for j,lab in enumerate(colorLabel):
                cbar3.ax.text(2.5,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center',color='hotpink')  #                         colorlist[0],colorLabel[0])

            plt.tight_layout()

            plt.show()
            plt.close()
    #add_Py3D_log(public_key_path,[glb_img_name,'\t',str(patchS),'\t',str(imageH),'\t',str(Longitudinal_Cracking_index//0.01*0.01),'\t',str(Transverse_Cracking_index//0.01*0.01),'\t',str(T_C_number//0.1*0.1),'\t'
    #    ,start_time.__str__(),'\t',datetime.now().__str__(),'\t',str(root.filename),'\n'])


def Repeat(glb_img_name,patchS=16):
    glb_file_path="D:/CentOS/GoogleEarthHighway_CNN/"+glb_img_name
    test_file_path="D:/CentOS/GoogleEarthHighway_U_net_crack255_early/"+glb_img_name

    if os.path.exists('D:/'):
        public_key_path='D:/CentOS/GoogleEarthHighway/'
    elif os.path.exists('/data/'):
        public_key_path='/data/GoogleEarthHighway/'

    Ortho_image = cv.imread(glb_file_path+"/"+glb_img_name+"Ortho_image.jpg",1)# 导入图片1/2H
    Ortho_image=cv.cvtColor(Ortho_image,cv.COLOR_BGR2RGB)
    imageH,imageW=Ortho_image.shape[0:2]
    Label_image=np.zeros((imageH,imageW))

    if os.path.exists(test_file_path+"/"+glb_img_name+'_512_U_Net_.csv'):
        PixelP=pd.read_csv(test_file_path+"/"+glb_img_name+'_512_U_Net_.csv',index_col=False,header=None)
        Label_image=np.array(PixelP).reshape((imageH,imageW)).copy()  # 2019.09.02
        Label_image[Label_image==255]=4
    else:
        print("error")

    print(glb_img_name,"\t",'Label_image Freq.',end='')
    ldic=array_element_counters(Label_image)

    colordic={'d':0,'c':4}

    try:
        PixelP=pd.read_csv(glb_file_path+"/"+glb_img_name+'_'+str(patchS)+'_Evaluation_Pred_Class.csv',index_col=False,header=None)
        #print('number of Pixels Loaded:',len(PixelP),len(PixelP[1]))
        Label_output=np.array(PixelP).reshape((imageH,imageW)).copy()  # 2019.09.02
        Label_output[Label_output!=4]=0
    except:
        print("error")

    #print('Label_image_output Successful! shape of Label_image:',Label_output.shape,end='')
    print(glb_img_name,"\t",'Label_image_output Freq.',end='')
    pdic=array_element_counters(Label_output)

    Ortho_image_back=Ortho_image.copy()
    Label_image_back=Label_image.copy()
    Label_output_back=Label_output.copy()

    colorlist=list(map(int,list(colordic.values())))
    colorLabel=list(map(str,list(colordic.keys())))
    dichavlist0=list(map(int,list(ldic.keys())))
    dichavlist1=list(map(int,list(pdic.keys())))
    listAll=colorlist#list(set(dichavlist0)&set(dichavlist1))
    listAll.sort()
    #print(glb_img_name,listAll)

    overall=True
    if overall:
        diff=(Label_image-Label_output)==0
        Ortho_image[diff]=255
        Label_image[diff]=255
        Label_output[diff]=255
        print(glb_img_name,'\t',patchS,'\tOverall\tPixel Accuracy:\t',np.sum(diff)/(imageW*imageH),'\t',np.sum(diff),'\tof\t',(imageW*imageH))

        imageH,imageW=Ortho_image.shape[0:2]
        fig=plt.figure("Comparing the Resutls_"+glb_img_name,figsize=(18,8))
        ax1=plt.subplot(2,2,3)

        ax1.imshow(Ortho_image)
        ax1.set_xlabel("Ortho_image")
        ax2=plt.subplot(2,2,1)
        im2=ax2.imshow(Label_image,vmin=-0.5,vmax=9.5,cmap='tab10')
        ax2.set_xlabel("Label_image")

        ax3=plt.subplot(2,2,2)
        im3=ax3.imshow(Label_output,vmin=-0.5,vmax=9.5,cmap='tab10')
        ax3.set_xlabel("Label_image_Patch-"+str(patchS))

        ax4=plt.subplot(2,2,4)
        im4=ax4.imshow(diff,vmin=0,vmax=1,cmap='gray')
        ax4.set_xlabel("Diff_Patch-"+str(patchS))

        #colordic={'wood':(10,0,0),'veg':(20,0,0),'sand':(30,0,0)}
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider2=make_axes_locatable(ax2)
        cax2=divider2.append_axes("right",size="3%",pad=0.05)  #colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
        cbar2=plt.colorbar(im2,cax=cax2,cmap='tab10')

        cbar2.set_ticks(colorlist)  # color bar
        cbar2.ax.get_yaxis().set_ticks([])
        for j,lab in enumerate(colorLabel):
            cbar2.ax.text(2.5,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center',color='hotpink')  #                         colorlist[0],colorLabel[0])
        divider3=make_axes_locatable(ax3)
        cax3=divider3.append_axes("right",size="3%",pad=0.05)  #colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
        cbar3=plt.colorbar(im3,cax=cax3,cmap='tab10')
        cbar3.set_ticks(colorlist)  # color bar
        cbar3.ax.get_yaxis().set_ticks([])
        for j,lab in enumerate(colorLabel):
            cbar3.ax.text(2.5,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center',color='hotpink')  #                         colorlist[0],colorLabel[0])

        colorlist2=[0,1]  #list(map(int,list(colordic.values())))
        colorLabel2=["Error","Matched"]  #list(map(str,list(colordic.keys())))
        divider4=make_axes_locatable(ax4)
        cax4=divider4.append_axes("right",size="3%",pad=0.05)  #colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
        cbar4=plt.colorbar(im4,cax=cax4,cmap='gray')
        cbar4.set_ticks(colorlist2)  # color bar
        cbar4.ax.get_yaxis().set_ticks([])
        for j,lab in enumerate(colorLabel2):
            cbar4.ax.text(2.5,colorlist2[j],"-"+str(colorlist2[j])+'- '+lab,ha='left',va='center',color='hotpink')  #                         colorlist[0],colorLabel[0])

        plt.tight_layout()
        #plt.show()
        plt.close()

    for i in listAll:

        Ortho_image=Ortho_image_back.copy()
        Label_image=Label_image_back.copy()
        Label_output=Label_output_back.copy()

        try:
            LT=(Label_image==i)
        except:
            #continue
            LT=Label_image.copy()
            LT[:,:]=False
        try:
            PT=(Label_output==i)
        except:
            #continue
            PT=Label_image.copy()
            PT[:,:]=False
        Label_image[~LT]=1000
        Label_output[~PT]=1001
        intersect=(Label_image-Label_output)==0
        blank=Label_image.copy()
        blank[:,:]=0
        blank[LT]=1
        blank[PT]=1
        union=(blank==1)
        Ortho_image[~union]=255
        Label_image[~LT]=255
        Label_output[~PT]=255
        if np.sum(union)!=0:
            print(glb_img_name,"\t",patchS,'\t',i,'\tIoU:\t',np.sum(intersect)/np.sum(union),'\t',np.sum(intersect),'\tof\t',np.sum(union),'\tLable:\t',np.sum(LT),'\tOutput:\t',np.sum(PT))
        else:
            print(glb_img_name,"\t",patchS,'\t',i,'\tIoU:\t','-\t',np.sum(intersect),'\tof\t',np.sum(union),'\tLable:\t',np.sum(LT),'\tOutput:\t',np.sum(PT))

        imageH,imageW=Ortho_image.shape[0:2]
        fig=plt.figure("Comparing the Resutls_"+glb_img_name,figsize=(18,8))
        ax1=plt.subplot(2,2,3)
        ax1.imshow(Ortho_image)
        ax1.set_xlabel("Union-"+str(np.sum(union))+'_'+str(i))

        ax2=plt.subplot(2,2,1)
        im2=ax2.imshow(Label_image,vmin=-0.5,vmax=9.5,cmap='tab10')
        ax2.set_xlabel("Label_image-"+str(np.sum(LT))+'_'+str(i))

        ax3=plt.subplot(2,2,2)
        im3=ax3.imshow(Label_output,vmin=-0.5,vmax=9.5,cmap='tab10')
        ax3.set_xlabel("Label_image_Patch-"+str(patchS)+"-"+str(np.sum(PT))+'_'+str(i))

        Ortho_image=Ortho_image_back.copy()
        Ortho_image[~intersect]=255
        ax4=plt.subplot(2,2,4)
        ax4.imshow(Ortho_image)
        ax4.set_xlabel("Intersection-"+str(np.sum(intersect))+'_'+str(i))

        #colordic={'wood':(10,0,0),'veg':(20,0,0),'sand':(30,0,0)}
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider2=make_axes_locatable(ax2)
        cax2=divider2.append_axes("right",size="3%",pad=0.05)  #colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
        cbar2=plt.colorbar(im2,cax=cax2,cmap='tab10')

        cbar2.set_ticks(colorlist)  # color bar
        cbar2.ax.get_yaxis().set_ticks([])
        for j,lab in enumerate(colorLabel):
            cbar2.ax.text(2.5,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center',color='hotpink')  #                         colorlist[0],colorLabel[0])
        divider3=make_axes_locatable(ax3)
        cax3=divider3.append_axes("right",size="3%",pad=0.05)  #colorlist=list(map(int,list(colordic.values()))) #cax=plt.axes([colorlist])
        cbar3=plt.colorbar(im3,cax=cax3,cmap='tab10')
        cbar3.set_ticks(colorlist)  # color bar
        cbar3.ax.get_yaxis().set_ticks([])
        for j,lab in enumerate(colorLabel):
            cbar3.ax.text(2.5,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center',color='hotpink')  #                         colorlist[0],colorLabel[0])

        plt.tight_layout()
        #plt.show()
        plt.close()

#------------------------------------------------------------------------------------------
patch=16
#Inputlist=['8','9','11','12','13']
#for I in Inputlist:
#    Repeat(I,patchS=patch)
GUIshowSINGLE(patchS=patch)