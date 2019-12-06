# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 01:46:24 2019

@author: 이해빈
"""

# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import cv2
import pafy
from skimage.measure import compare_ssim as ssim
import os
import scipy.stats
import time

def mse(imageA, imageB):
    imageA=cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB=cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    err=np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    return err

def hist_bhattacharyya(imageA, imageB):
    hsv1=cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    hsv2=cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
    
    hist1=cv2.calcHist([hsv1], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    hist2=cv2.calcHist([hsv2],[0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    
    value=cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    
    return value

def extraction(url):
    
    vPafy = pafy.new(url)
    play = vPafy.getbest(preftype="webm")
    cap = cv2.VideoCapture(play.url)

    count = 0
    thresh_hist = 0.1
    thresh_bha = 0.5
    thresh_mse = 2000
    thresh_ssim = 0.45

    ret, prev_frame = cap.read()
    cv2.imwrite('frame%d.jpg'%count,prev_frame);
    count = 1

    while ret:
        ret, curr_frame = cap.read()

        hist_prev = cv2.calcHist([prev_frame], [0], None, [256], [0,256])
        hist_prev = cv2.normalize(hist_prev, hist_prev).flatten()
        hist = cv2.calcHist([curr_frame], [0], None, [256], [0,256])
        hist = cv2.normalize(hist, hist).flatten()
        
        if ret:
            d = dist.chebyshev(hist_prev, hist)
            if d > thresh_hist:          
                value_b=hist_bhattacharyya(prev_frame, curr_frame)
                if value_b>thresh_bha:
                    error=mse(prev_frame,curr_frame)
                    if error>thresh_mse:
                        grayP=cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        grayC=cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                        (score, diff)=ssim(grayP, grayC, full=True)
                        if score<thresh_ssim:
                            cv2.imwrite('frame%d.jpg'%count,curr_frame);
                            count += 1
                            prev_frame = curr_frame                                        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    print(count)
        
def delete(path):
    for root, dirs, files in os.walk(path):
        for currentFile in files:
            exts = ('.png', '.jpg')
            if currentFile.lower().endswith(exts):
                os.remove(os.path.join(root, currentFile))   

url=input("url : ")
start_time = time.clock()
extraction(url)
print("--- %s seconds ---" % (time.clock() - start_time))
