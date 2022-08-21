#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 15:57:02 2022

@author: sergey
"""
import GRZ_mod_CV2_bigNet as grz

import os
#import matplotlib.pyplot as plt
#from tensorflow.keras.preprocessing import image


import cv2


nums = []

lag_dir = '/home/sergey/AFE/ГРЗ/Ложняки/front/train/'

grz_ = grz.GRZ(lag_dir)

ld = os.listdir(lag_dir)

for fname in ld:
    
    grz_.Get_segments_from_pictue (fname)
    nums = grz_.Get_GRZ_from_segments()
    
    print (nums)    
