#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 15:57:02 2022

@author: sergey
"""
import GRZ_mod_CV2_bigNet as grz

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


import cv2


nums = []

lag_dir = '/home/sergey/AFE/ГРЗ/Ложняки/front/train/'

grz_ = grz.GRZ(lag_dir)

ld = os.listdir(lag_dir)

for fname in ld:
    
    grz_.Get_segments_from_pictue (fname)
    nums = grz_.Get_GRZ_from_segments()
    
    frame = cv2.imread(lag_dir+fname)
    
    for rec, num in zip(grz_.coords,nums):
        
        cv2.rectangle(frame,(rec[0],rec[1]), (rec[2],rec[3]),(255,255,255),1)
        position = (rec[0],rec[1]-50)
    
    
        cv2.putText(
           frame, #numpy array on which text is written
           num, #text
           position, #position at which writing has to start
           cv2.FONT_HERSHEY_SIMPLEX, #font family
           1, #font size
           (255,255,255),) #font color)
    
   
    cv2.imshow('frame',frame)
    
    cv2.waitKey()
    
    
