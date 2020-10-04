#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
# Copyright:    Yuhan Jiang
# Email:        yuhan.jiang@marquette.edu
# Date:         10/03/2020
# Discriptions : Read the GPS from Screenshot of Google Earth web (fullscreen on 2nd monitor)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
import cv2 as cv
import re
import matplotlib.pyplot as plt

# read image
def ORC_GoogleEarthScreenshotGPS(imgName,path,showResult_bool=True):
    im = cv.imread(path+'Capture'+str(imgName)+'.PNG')
    img_rgb = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    part=[]
    part.append(img_rgb[1050:1079,3593:3618])#camera
    part.append(img_rgb[1050:1079,3641:3655])#N_degree
    part.append(img_rgb[1050:1079,3658:3674])#N_min
    part.append(img_rgb[1050:1079,3677:3691])#N_sec
    part.append(img_rgb[1050:1079,3707:3721])#W_degree
    part.append(img_rgb[1050:1079,3726:3740])#W_min
    part.append(img_rgb[1050:1079,3743:3756])#W_sec

    partcopy=part.copy()

    StringList=['Camera','N_deg','N_min','N_sec','W_deg','W_min','W_sec']

    letter=[]
    for i in range(len(part)):
        part[i]=cv.bilateralFilter(part[i],-1,5,5)
        ret,part[i]=cv.threshold(part[i],160,255,cv.THRESH_BINARY_INV)

        lt=re.sub('[\W_]+','',pytesseract.image_to_string(part[i]))

        if 'a' in lt:# handel the error 8 be detected as a
            new=""
            for i in range(len(lt)):
                if i==lt.index('a'):
                    new=new+'8'
                else:
                    new=new+lt[i]
            lt=new

        if lt=='' or lt.isdigit()==False: # check blank and non- number
            imagebackup=partcopy[i]# try second time, otherwise manually input it
            imagebackup=cv.bilateralFilter(imagebackup,-1,5,5)
            ret2,imagebackup=cv.threshold(imagebackup,150,255,cv.THRESH_BINARY)

            lt=re.sub('[\W_]+','',pytesseract.image_to_string(imagebackup))

            if lt=='' or lt.isdigit()==False:  # check blank and non- number
                plt.figure("Capture"+str(imgName)+'_'+StringList[i]+'_Err!_'+str(lt))
                plt.imshow(imagebackup)
                plt.show()
                lt=input("Err! Please Input the Number in the Figure:")
                plt.clf()
                plt.close()

        if lt.isdigit():
            letter.append(lt)
        else:
            print("Error")


            #print("Camera:",letter[0],end='\t')
            #print(letter[1],"°",letter[2],"'",letter[3],'" N',end='\t')
            #print(letter[4],"°",letter[5],"'",letter[6],'" W',end='\t')

    if showResult_bool:

            plt.figure("Capture"+str(imgName))
            plt.ion()
            ax=[]
            for i in range(len(part)):
                ax.append(plt.subplot(1,int(len(part)),int(i)+1))
                ax[i].set_xlabel(letter[i])
                plt.imshow(part[i])
            plt.pause(3)
            plt.ioff()
            plt.clf()
            plt.close()
    print(str(imgName),'\t',"Camera:",letter[0],end='\t')
    print(letter[1],"°",letter[2],"'",letter[3],'" N',end='\t')
    print(letter[4],"°",letter[5],"'",letter[6],'" W',end='\n')
    return letter,StringList
#-----------------
imglist=range(50)
path='D:/CentOS/G2/'
for i in imglist:
    ORC_GoogleEarthScreenshotGPS(i,path)