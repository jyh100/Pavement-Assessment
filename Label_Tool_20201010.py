#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
# Copyright:    Yuhan Jiang
# Email:        yuhan.jiang@marquette.edu
# Date:         10/10/2020
# major update:    Screenshot tool save ortho_image, open the saved file instead of Caputre.PNG
# previous update: Label the Ortho-image and save the label anc color name;color bar;input un-pre-defined-num-point;public key; save label image as .csv file
# next update:
#
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

def press(event):
    global Label_image
    sys.stdout.flush()
    global vexNum
    global mouseloc
    global colordic,colorlist,colorLabel
    global lastsetp_Label_image
    vexNum=len(mouseloc)
    if event.key=='c' and vexNum>=2:
            print('Draw Crack Line ShortCut Selected ',vexNum, " Points")
            pointarray=np.array(mouseloc,np.int32)
            print(pointarray)  # print selected points
            xlist=[]
            ylist=[]
            for i in range(len(pointarray)):  # get the lenght of
                xlist.append(int(pointarray[i][0]))
                ylist.append(int(pointarray[i][1]))
            nameOfArea='c'# or nameOfArea=='crack':
            lastsetp_Label_image=Label_image.copy()
            cv.polylines(Label_image,[pointarray],False,colordic[nameOfArea],thickness=12)#update to line with withd
            for i in range(len(pointarray)):  # draw the selected point
                    ax1.scatter(xlist[i],ylist[i],marker='+',color='hotpink',s=35)
            ax1.plot(xlist,ylist,color='pink',linewidth=5)#update to line with width
            ax1.text(sum(xlist)/len(xlist),sum(ylist)/len(ylist),nameOfArea)
            ax2.text(sum(xlist)/len(xlist),sum(ylist)/len(ylist),nameOfArea)
            ax2.imshow(Label_image,vmin=-0.5,vmax=9.5,cmap='tab10')
            ax2.set_xlabel("Label_image \n New Object:"+nameOfArea+" Labeled",color='b')
            colorlist=list(map(int,list(colordic.values())))
            colorLabel=list(map(str,list(colordic.keys())))
            cbar.set_ticks(colorlist)# color bar
            cbar.ax.get_yaxis().set_ticks([])
            for j,lab in enumerate(colorLabel):
                    cbar.ax.text(2.5,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center',color='b')  #                         colorlist[0],colorLabel[0])
            cbar.ax.get_xaxis().labelpad=15  #distance between axis label and tick label
            cbar.ax.set_xlabel('Defined Object #'+str(len(colorlist)))
            fig.canvas.draw()
            print("Labeled the object",nameOfArea,'=',colordic[nameOfArea],"in the Image")
            #print('Label_image Freq.')
            #array_element_counters(Label_image)
            fig.canvas.draw()
            mouseloc.clear()
            vexNum=len(mouseloc)
            ax1.set_xlabel("Google Earth Imagery \n Select Points here",color='black')
            fig.canvas.draw()

    elif event.key=='d':
        if vexNum>2:
            print('Draw Object Area with Selected ',vexNum, " Points")
            pointarray=np.array(mouseloc,np.int32)
            print(pointarray)  # print selected points
            xlist=[]
            ylist=[]
            for i in range(len(pointarray)):  # get the lenght of
                xlist.append(int(pointarray[i][0]))
                ylist.append(int(pointarray[i][1]))
            ax1.scatter(xlist[0],ylist[0],marker='+',color='cyan',s=80)
            for i in range(1,len(pointarray)):# draw the selected point
                ax1.scatter(xlist[i],ylist[i],marker='+',color='r',s=30)
            ax1.set_xlabel("Google Earth Imagery \n Input the object name in console\n Input 'n' for skip ",color='green')
            fig.canvas.draw()

            #print(xlist,ylist)
            #print('Do you use the select',len(vexNum),'points?')
            print("Predefined object and color",colordic)
            nameOfArea=input('Input object Name to use the select '+str(len(pointarray))+' Points, otherwise Input n to skip them: ')
            if nameOfArea not in colordic and nameOfArea!='n':
                newcolor=int(input("Define NEW Object Color/Labels: "))
                if newcolor in colordic.values():
                    newcolor=int(input("Existed Color/Labels, Input New Color/Labels: "))
                colordic[nameOfArea]=newcolor
                print("Defined New Object",nameOfArea,'=',colordic[nameOfArea])
            if nameOfArea!='n' and nameOfArea!='c':
                lastsetp_Label_image=Label_image.copy()
                cv.fillPoly(Label_image,[pointarray],colordic[nameOfArea])
                for i in range(len(pointarray)):  # draw the selected point
                    ax1.scatter(xlist[i],ylist[i],marker='+',color='cyan',s=35)
                ax1.fill(xlist,ylist)
                ax1.text(sum(xlist)/len(xlist),sum(ylist)/len(ylist),nameOfArea)
                ax2.text(sum(xlist)/len(xlist),sum(ylist)/len(ylist),nameOfArea)
                ax2.imshow(Label_image,vmin=-0.5,vmax=9.5,cmap='tab10')
                ax2.set_xlabel("Label_image \n New Object:"+nameOfArea+" Labeled",color='b')
                colorlist=list(map(int,list(colordic.values())))
                colorLabel=list(map(str,list(colordic.keys())))
                cbar.set_ticks(colorlist)# color bar
                cbar.ax.get_yaxis().set_ticks([])
                for j,lab in enumerate(colorLabel):
                    cbar.ax.text(2.5,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center',color='b')  #                         colorlist[0],colorLabel[0])
                cbar.ax.get_xaxis().labelpad=15  #distance between axis label and tick label
                cbar.ax.set_xlabel('Defined Object #'+str(len(colorlist)))
                fig.canvas.draw()
                print("Labeled the object",nameOfArea,'=',colordic[nameOfArea],"in the Image")
                #print('Label_image Freq.')
                #array_element_counters(Label_image)
            elif nameOfArea=='c' or nameOfArea=='crack':
                lastsetp_Label_image=Label_image.copy()
                cv.polylines(Label_image,[pointarray],False,colordic[nameOfArea],thickness=12)#update to line with withd
                for i in range(len(pointarray)):  # draw the selected point
                    ax1.scatter(xlist[i],ylist[i],marker='+',color='hotpink',s=35)
                ax1.plot(xlist,ylist,color='pink',linewidth=5)#update to line with width
                ax1.text(sum(xlist)/len(xlist),sum(ylist)/len(ylist),nameOfArea)
                ax2.text(sum(xlist)/len(xlist),sum(ylist)/len(ylist),nameOfArea)
                ax2.imshow(Label_image,vmin=-0.5,vmax=9.5,cmap='tab10')
                ax2.set_xlabel("Label_image \n New Object:"+nameOfArea+" Labeled",color='b')
                colorlist=list(map(int,list(colordic.values())))
                colorLabel=list(map(str,list(colordic.keys())))
                cbar.set_ticks(colorlist)# color bar
                cbar.ax.get_yaxis().set_ticks([])
                for j,lab in enumerate(colorLabel):
                    cbar.ax.text(2.5,colorlist[j],"-"+str(colorlist[j])+'- '+lab,ha='left',va='center',color='b')  #                         colorlist[0],colorLabel[0])
                cbar.ax.get_xaxis().labelpad=15  #distance between axis label and tick label
                cbar.ax.set_xlabel('Defined Object #'+str(len(colorlist)))
                fig.canvas.draw()
                print("Labeled the object",nameOfArea,'=',colordic[nameOfArea],"in the Image")
                #print('Label_image Freq.')
                #array_element_counters(Label_image)
            elif nameOfArea=='n':
                for i in range(len(pointarray)):  # draw the selected point
                    ax1.scatter(xlist[i],ylist[i],marker='+',color='y',s=30)
        print("Press Any Key to continue")
        fig.canvas.draw()
        mouseloc.clear()
        vexNum=len(mouseloc)
        print("Please select<",MaxNum,"Points as Vertexes of a Object,Then Press Key 'd' for Drawing, or Press Key '0' to exit App")
        ax1.set_xlabel("Google Earth Imagery \n Select Points here",color='black')
        fig.canvas.draw()

    elif event.key == '0': # close
            boolClose=input("Ready to close App y/n?")
            if boolClose=='y':
                global endloopbool
                endloopbool=0
                cv.imwrite(glb_file_path+"/"+glb_img_name+'Label_image.jpg',Label_image*255/10,[int(cv.IMWRITE_JPEG_QUALITY),100])
                data=pd.DataFrame(Label_image)  #pandas DataFrame。
                data.to_csv(glb_file_path+"/"+glb_img_name+'Label_image.csv',index=False,header=False)
                print("Saved label image")
                try:
                    with open(public_key_path+'Public_Object_Label_Dic.csv','w') as f:
                        w=csv.writer(f,lineterminator='\n')
                        for key,val in colordic.items():
                            w.writerow([key,val])
                    print('Saved label csv file')
                except IOError:
                    print("I/O error")
                mouseloc.clear()
                vexNum=len(mouseloc)
                #root.destroy()
                plt.waitforbuttonpress()
                ax1.cla()
                ax2.cla()
                #plt.close(fig)                  #close the windows

            elif boolClose=='n': #
                print('Continue Pick up Points')
                cv.imwrite(glb_file_path+"/"+glb_img_name+'Label_image.jpg',Label_image*255/10,[int(cv.IMWRITE_JPEG_QUALITY),100])
                data=pd.DataFrame(Label_image)  #pandas DataFrame。
                data.to_csv(glb_file_path+"/"+glb_img_name+'Label_image.csv',index=False,header=False)
                print("Saved label image")
                try:
                    with open(public_key_path+'Public_Object_Label_Dic.csv','w') as f:
                        w=csv.writer(f,lineterminator='\n')
                        for key,val in colordic.items():
                            w.writerow([key,val])
                    print('Saved label csv file')
                except IOError:
                    print("I/O error")
            #fig.canvas.draw()
    elif event.key=='s': # save
        cv.imwrite(glb_file_path+"/"+glb_img_name+'Label_image.jpg',Label_image*255/10,[int(cv.IMWRITE_JPEG_QUALITY),100])
        #save label image 2019.09.02
        data=pd.DataFrame(Label_image)  #pandas DataFrame。
        data.to_csv(glb_file_path+"/"+glb_img_name+'Label_image.csv',index=False,header=False)
        print("Saved label image")
        try:
            with open(public_key_path+'Public_Object_Label_Dic.csv','w') as f:
                w=csv.writer(f,lineterminator='\n')
                for key,val in colordic.items():
                    w.writerow([key,val])
            print('Saved label csv file')
        except IOError:
            print("I/O error")
    elif event.key=='n': #new label
        ax2.cla()
        Label_image[:]=255
        print("Created New Label_image")
        print('Label_image Freq.')
        array_element_counters(Label_image)
        ax2.set_xlabel("Label_image \n Redo Label Process",color='red')
        ax2.imshow(Label_image,vmin=-0.5,vmax=9.5,cmap='tab10')
        print("Label_image: \t Redo Label Process")
        ax1.cla()
        ax1.imshow(Ortho_image)
    elif event.key=='z': #return the last setp
        Label_image=lastsetp_Label_image.copy()
        print("Label_image: \t Cleaning")
        ax2.cla()
        print("Return to last step")
        print('Label_image Freq.')
        array_element_counters(Label_image)
        ax2.set_xlabel("Label_image \n Undo Label the last Process",color='red')
        ax2.imshow(Label_image,vmin=-0.5, vmax=9.5,cmap='tab10')
        print("Label_image: \t Undo Label the last Process")
        ax1.cla()
        ax1.imshow(Ortho_image)


    elif event.key=='l': #cls
        print("Label_image: \t Cleaning")
        ax1.cla()
        ax1.imshow(Ortho_image)

    elif event.key=='a': #statistic ortho-image
        print('Class_image Freq.')
        array_element_counters(Label_image)
        print(colordic)
        print(array_element_mode(Label_image))
        num_bins=10
        fig2=plt.figure("Frequence Summary"+glb_img_name,figsize=(6,3))
        plt.cla()
        n,bins,patches=plt.hist(Label_image.reshape((-1,1)),num_bins,color='blue')
        plt.xlim([-0.5,9.5])
        plt.xticks(colorlist)
        plt.ylim([0,1000])
        fig2.show()

    mouseloc.clear()
    vexNum=len(mouseloc)
    fig.canvas.draw()
# 点击，显示，程序
def click_get_profile():
    global vexNum
    global mouseloc
    #plt.clf()
    mouseloc = plt.ginput(MaxNum,timeout=-1) # never timeout
    vexNum=len(mouseloc)
    if vexNum<2:
        ax1.set_xlabel("Google Earth Imagery \n Select "+str(vexNum)+", while need at least 2 Points here",color='black')
        mouseloc.clear()
        vexNum=len(mouseloc)
        #print(vexNum,'Point Selected')
    elif vexNum==2:
        ax1.set_xlabel("Google Earth Imagery \n Select # Points:"+str(vexNum)+" Press C to Draw Crack",color='pink')
    elif vexNum>2:
        ax1.set_xlabel("Google Earth Imagery \n Select # Points:"+str(vexNum)+" Press C/D to Draw",color='blue')
    fig.canvas.mpl_connect('key_press_event',press)

        #fig.canvas.mpl_connect('button_press_event',click)
    '''
    pointarray=np.array(mouseloc,np.int32)
    xlist=[]
    ylist=[]
    for i in range(len(pointarray)):  # get the lenght of
        xlist.append(int(pointarray[i][0]))
        ylist.append(int(pointarray[i][1]))
    for i in range(len(pointarray)):
        ax1.scatter(xlist[i],ylist[i],marker='x',color='red',s=40)
    '''

root = tk.Tk()
root.title("Label_image V0.01")#Please Open A Ortho image
root.geometry("300x300")
theLabel=tk.Label(root,text='Author: Yuhan Jiang')
theLabel.pack()#用于自动调节组件自身的尺寸
theLabel=tk.Label(root,text='E-mail: yuhan.jiang@marquette.edu')
theLabel.pack()#用于自动调节组件自身的尺寸
theLabel=tk.Label(root,text='Step 1:Please open an image')
theLabel.pack()#用于自动调节组件自身的尺寸
#root.mainloop()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpg files","*Ortho_image.jpg"),("all files","*.*")))
print ("Opened",root.filename)
glb_file_path,file_name=get_filePath_fileName_fileExt(root.filename)[0:2]
glb_img_name =file_name[:-11]#print(glb_img_name)

if os.path.exists('D:/'):
    public_key_path='D:/CentOS/GoogleEarthHighway/'
elif os.path.exists('/data/'):
    public_key_path='/data/GoogleEarthHighway/'

print('glb_file_path,file_name,public_key_path',glb_file_path,file_name,public_key_path)

# 载入input图像@EHENCED_Elevation_map
Ortho_image = cv.imread(root.filename,1)# 导入图片1/2H
print("Highway image Successful!")
#Ortho_image=Ortho_image[0:1024,252:252+1408,:]#crop the captured image
#cv.imwrite(glb_file_path+"/"+glb_img_name+'Ortho_image.jpg',Ortho_image,[int(cv.IMWRITE_JPEG_QUALITY),100])
imageH,imageW=Ortho_image.shape[0:2]
print("Ortho-image,",glb_img_name,imageH,imageW)
Ortho_image=cv.cvtColor(Ortho_image,cv.COLOR_BGR2RGB)
Label_image=np.zeros((imageH,imageW))

if os.path.exists(glb_file_path+"/"+glb_img_name+'Label_image.csv'):
    # read label image 2019.09.02
    PixelP=pd.read_csv(glb_file_path+"/"+glb_img_name+'Label_image.csv',index_col=False,header=None)
    print('number of Pixels Loaded:',len(PixelP),len(PixelP[1]))
    Label=np.array(PixelP).reshape((imageH,imageW)).copy()  # 2019.09.02
    Label_image=np.zeros((imageH,imageW))
    Label_image[:]=Label[:imageH,:imageW]
    Label_image[Label_image==6]=4
    print('Label_image Successful! shape of Label_image:',Label_image.shape)
else:
    Label_image[:]=0# defualt
    print("Created New Label_image")

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

theLabel=tk.Label(root,text='Step 2: Please Use the mouse to pick up points')
theLabel.pack()
theLabel=tk.Label(root,text='Click X on windows to end Label_Ortho_image App')
theLabel.pack()
root.destroy()
imageH,imageW=Ortho_image.shape[0:2]

fig=plt.figure("Label_image_"+glb_img_name,figsize=(18,8))
ax1=plt.subplot(1,2,1)
plt.xlim([-10,imageW+10])
plt.ylim([imageH+10,-10])
ax1.imshow(Ortho_image)
ax1.set_xlabel("Google Earth Imagery \n Select Point Here \n Left Click to add, Right Click to remove the last one, Middle Click to Complet Pickup Point")
ax2=plt.subplot(1,2,2)
im=ax2.imshow(Label_image,vmin=-0.5,vmax=9.5,cmap='tab10')
plt.xlim([-10,imageW+10])
plt.ylim([imageH+10,-10])
ax2.set_xlabel("Label_image")
vexNum=4
mouseloc=[]
MaxNum=200
#colordic={'wood':(10,0,0),'veg':(20,0,0),'sand':(30,0,0)}
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

print("Please select<",MaxNum," Points as Vertexes of a Object,Then Press Key 'D' for Drawing")
print("Left Click to add, Right Click to remove the last one, Middle Click to Complet Pickup Point")
endloopbool=1
while endloopbool:
    click_get_profile()# call click function
    #if plt.

plt.show()

check=pd.read_csv(glb_file_path+"/"+glb_img_name+'Label_image.csv',index_col=False,header=None)
print('number of Pixels Loaded:',len(check),len(check[1]))
check_image=np.array(check).reshape((imageH,imageW)).copy()  # 2019.09.02
array_element_counters(check_image)
###################add
