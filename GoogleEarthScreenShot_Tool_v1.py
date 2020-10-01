#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
# Copyright:    Yuhan Jiang
# Email:        yuhan.jiang@marquette.edu
# Date:         10/01/2020
# Discriptions : Tracking roadway on Google Earth web (fullscreen on 2nd monitor), 
import math
import statistics
from PIL import ImageGrab
import time
from scipy.ndimage import filters
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from pynput.keyboard import Key, Controller

def get_median(listt):
    listt=list(map(float,listt))
    return float(statistics.median(listt))
def get_mean(listt):
    listt=list(map(float,listt))
    return float(statistics.mean(listt))
def line_regression(x,y):
    n=np.size(x)

    x_mean=np.mean(x)
    y_mean=np.mean(y)
    x_mean,y_mean

    Sxy=np.sum(x*y)-n*x_mean*y_mean
    Sxx=np.sum(x*x)-n*x_mean*x_mean

    b1=Sxy/Sxx
    b0=y_mean-b1*x_mean
    #print('slope b1 is',b1)
    #print('intercept b0 is',b0)
    return b0,b1

keyboard=Controller()
time.sleep(2)# wait 2 sec to start screensnap, click on 2nd monitor (if used), where the Google Earth web is opened with fullscreen
keyboard.press('u')# switch to google earth to top-view.
keyboard.release('u')
keyboard.press('n')# head to north
keyboard.release('n')

for i in range(50):# capture fifty images
    angle=15
    x_shift=100
    angleB=3 #angle diff. boundary
    x_shiftB=15 # x-direction diff. boundary
    sideB=200 #  margin boundary
    num_change=0 # number of changes in each station
    while abs(angle)>angleB or abs(x_shift)>x_shiftB:
            time.sleep(.5)  # wait to load the map
            image = ImageGrab.grab(all_screens=True)#image = ImageGrab.grab(bbox=(0,0,700,800))
            imShot = image.crop((1920+252, 0, 1920+252+1408, 1024))#remove tool bar areas
            RGB=np.array(imShot)
            imGray=cv.cvtColor(RGB,cv.COLOR_RGB2GRAY) # convert to grayscale
            gray_filtered=cv.bilateralFilter(imGray,7,50,50)            # Smooth without removing edges.
            gray_filtered[gray_filtered<25]=255#keep the shade, sealed crack, crack, black vehicels
            gray_filtered[gray_filtered<150]=0# keep the pavement surface, lane marks
            #edges_filtered=cv.Canny(gray_filtered,60,120,L2gradient=True)            # Apply the canny filter
            y_bar=np.zeros((1024,1408))
            filters.sobel(gray_filtered,1,y_bar)                #Apply the sobel filter to get y-direction edge
            y_bar[y_bar<000]=0                                  #remove the negtive value
            y_bar[:,:sideB]=0                                   # remove margin
            y_bar[:,1408-sideB:]=0                              # remove margin

            xlist=[]
            ylist=[]
            for yi in range (1024):
                try:
                    top=np.where(y_bar[yi,]>000)[0]#600:800
                    xi=get_mean(top)
                    xlist.append(int(xi))
                    ylist.append(yi)
                except:
                    pass

            y_p=np.array(ylist)
            x_p=np.array(xlist)
            b0,b1 = line_regression(y_p,x_p)# x=b0+b1*y
            xlist_predict=[]
            ylist_predict=[]
            for yi in range (1024):
                xlist_predict.append(b0+b1*yi)
                ylist_predict.append(yi)

            x1=xlist_predict[0]
            x2=xlist_predict[-1]
            angle=math.atan((x2-x1)/1024)/math.pi*180
            x_shift=(x1+x2)/2-1408/2  #
            print(x1,x2,angle,x_shift)

            if abs(angle)>35 or num_change>=5: # change the direction to North head
                keyboard.press('n')
                keyboard.release('n')
                print('head north')
                num_change+=1
            elif angle>angleB: # slighlty rotate the direction
                with keyboard.pressed(Key.shift):
                    keyboard.press(Key.right)
                    time.sleep(0.05)
                    keyboard.release(Key.right)
                print('shift+right')
                num_change+=1
            elif angle<-angleB: # slighlty rotate the direction
                with keyboard.pressed(Key.shift):
                    keyboard.press(Key.left)
                    time.sleep(0.05)
                    keyboard.release(Key.left)
                print('shift+left')
                num_change+=1
            elif x_shift>x_shiftB:
                keyboard.press(Key.right)
                time.sleep(0.08)
                keyboard.release(Key.right)
                print('right')
                num_change+=1
            elif x_shift<-x_shiftB:
                keyboard.press(Key.left)
                time.sleep(0.08)
                keyboard.release(Key.left)
                print('left')
                num_change+=1
            if num_change>=5:
                print('Error!')
                num_change=0#reset
                time.sleep(2)

    if True: # show images

            fig=plt.figure("HighwayASS_Project"+str(i),figsize=(12,6))
            plt.ion()  #turn on interaction mode
            ax1=plt.subplot(1,2,1)
            ax1.imshow(y_bar,cmap='gray')
            ax1.plot(xlist_predict,ylist_predict,color='pink',linewidth=5)#update to line with width
            ax2=plt.subplot(1,2,2)
            ax2.plot(xlist_predict,ylist_predict,color='pink',linewidth=5)#update to line with width
            ax2.imshow(gray_filtered,cmap='gray')
            plt.pause(1.5)# show image for 1.5 sec plt.show()            time.sleep(2)
            plt.ioff()#turn off interaction mode, avoid
            plt.clf()#clean image
            plt.close(fig)#close window

    imShot.save('D:/CentOS/G/Capture'+str(i)+'.png') # save images

    keyboard.press(Key.up)
    time.sleep(1.9)#move to next image, 1.8 sec move up about 1024 pixels (scroll up one screen)
    keyboard.release(Key.up)
