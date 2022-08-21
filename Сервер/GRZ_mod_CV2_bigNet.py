# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:42:55 2022

@author: sergey
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import re
import cv2

Y_pict = 240
X_pict = 480

Y_num = 50
X_num = 144

# Определяем метрику и функцию потерь

def IoU(y_true, y_pred):
	y_true_f = tf.keras.backend.flatten(y_true)
	y_pred_f = tf.keras.backend.flatten(y_pred)
	intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
	return (2. * intersection + 1.0) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1.0)


class GRZ():
    
    def __init__(self,work_dir):
        self.frame_network = models.load_model('/home/sergey/AFE/ГРЗ/Сети/best_net_big04_8966_85577.h5',custom_objects={'IoU': IoU})
        self.GRZ_network = models.load_model('/home/sergey/AFE/ГРЗ/Сети/Symbol_net_8344.h5',custom_objects={'IoU': IoU})
        self.GRZ_segments = []
        self.mixer = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'a': 10,
        'b': 11, 'c': 12, 'e': 13, 'k': 14, 'm': 15, 'n': 16, 'o': 17, 'p': 18, 't': 19, 'x': 20, 'y': 21 }
        self.demixer = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'а',
        11: 'в', 12: 'с', 13: 'е', 14: 'к', 15: 'м', 16: 'н', 17: 'о', 18: 'р', 19: 'т', 20: 'х', 21: 'у'}
        self.symbol_tr = 74 # граница площади символа в предсказанной маске
        self.file_name = ''
        self.image_path = work_dir
        self.coords = []
        
    
    
    def Get_segments_from_pictue(self, fname):
        
        im = image.load_img(self.image_path+fname, target_size = (Y_pict,X_pict))
        fim = image.load_img(self.image_path+fname)
        fim = image.img_to_array(fim)


        im = image.img_to_array(im)
       # im = im[20:,:,:] # понижаю размерность для лучшего прохода через UNET
        im = im/255

        im = np.expand_dims(im, axis = 0)

        pred = self.frame_network.predict(im)
       
        tr = (pred.reshape((Y_pict,X_pict))>0.5).astype('uint8')
        cont, heir = cv2.findContours(tr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.GRZ_segments = []
        self.coords = []
        for c in cont:
            x,y, w,h = cv2.boundingRect(c)
    
            numb = fim[int(round(y*fim.shape[0]/Y_pict)):int(round((y+h)*fim.shape[0]/Y_pict)),int(round(x*fim.shape[1]/X_pict)):int(round((x+w)*fim.shape[1]/X_pict)) ]
 
            self.GRZ_segments += [cv2.resize(numb,(X_num,Y_num))]
            
            self.coords.append([int(round(x*fim.shape[1]/X_pict)),int(round(y*fim.shape[0]/Y_pict)), int(round((x+w)*fim.shape[1]/X_pict)),int(round((y+h)*fim.shape[0]/Y_pict))])
            
#            if numb.shape[0]>0:
#                numb = image.array_to_img(numb)
#            
 #               numb = numb.resize((X_num,Y_num))
 #           
 #               numb = image.img_to_array(numb)
            
#                self.GRZ_segments += [numb]
            

    def Get_GRZ_from_segments(self):
        
        GRZs = []
        if len(self.GRZ_segments) > 0:
            for im in self.GRZ_segments:
                im = im[:-2,:,:] # понижаю размерность для лучшего прохода через UNET 
                im = im/255
                im = np.expand_dims(im, axis=0)

                pred = self.GRZ_network.predict(im)
                pred = pred.reshape(48,144,23)
                number_cucumber = {}

                GRZ = ''
                for i in range(22):
    
                    tr = ((pred[:,:,i] - pred[:,:,22])>0.5).astype('uint8') 
                    cont, heir = cv2.findContours(tr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    #cont = Get_contours_symb(tr)
                    for c in cont:
                        x,y, w,h = x,y, w,h = cv2.boundingRect(c)
                        if w*h > self.symbol_tr:
                            number_cucumber[x] = self.demixer[i]
#                            if w < 0.9*h:
#                                number_cucumber[x] = self.demixer[i]
#                            else:
#                                number_cucumber[x] = self.demixer[i]
#                                number_cucumber[x+5] = self.demixer[i]

                sk = sorted(number_cucumber.keys())

                for i in sk:
                    GRZ = GRZ + number_cucumber[i]
               

                match = re.search(r'\D\d\d\d\D\D\d\d+', GRZ) 
                mainRF = match[0] if match else ''

                match = re.search(r'\D\D\d\d\d\d\d+', GRZ) 
                taxi = match[0] if match else ''
                
                match = re.search(r'\D\d\d\d\d\d\d+', GRZ) 
                police = match[0] if match else ''
                
                if mainRF != '':
                    GRZs += [mainRF]   

                if taxi != '':
                    GRZs += [taxi]   

                if police != '':
                    GRZs += [police]   


        return GRZs
                
   
    
    def Show_segs(self):
        plt.imshow(self.GRZ_segments[0])
        plt.show()
        
    def get_segments_len(self):
        return len(self.GRZ_segments)
        
        
        
        
