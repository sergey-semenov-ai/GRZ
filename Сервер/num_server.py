#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:26:09 2019

@author: sergey
"""

from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from http.server import HTTPServer, BaseHTTPRequestHandler
import uuid
import base64
import json
import numpy as np
import os, time
import GRZ_mod_CV2_bigNet as GRZ


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def _text(self, message):
        """This just generates an HTML document that includes `message`
        in the body. Override, or re-write this do do more interesting stuff.
        """
        content = f"{message}"
        return content.encode("utf8")  # NOTE: must return a bytes object!    

    def _set_headers(self):
        self.send_response(200)
        #self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
    def do_GET(self):
        self._set_headers()
       # self.wfile.write(b'Hello, world!')
    def do_POST(self):
        self._set_headers()
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        filename = 'images/'+str(uuid.uuid4())+'.jpg' # генеррируются уникальные имена файлов
        base64_data = base64.b64decode(body) # собственно декодирование
        with open(filename, 'wb') as f:
            f.write(base64_data)
            
        nums = []

        t0 = time.time()

        
        decoder.Get_segments_from_pictue(filename)
        
        segments_found = decoder.get_segments_len()
        
        nums = decoder.Get_GRZ_from_segments()

        ans = {'segments_found': str(segments_found)}

        for i in range (len(nums)):
           
            ans[str(i)] = nums[i]

        self.wfile.write(bytes(json.dumps(ans, ensure_ascii=False), 'utf-8'))
        
        os.remove(filename)
        
        t1 = time.time()
        print (t1-t0) 

    def do_HEAD(self):
        self._set_headers()

PORT=8000

# загружаю обученную нейронную сеть

#Binary = models.load_model('bi_1412_VGG_9661_9197.h5')

decoder = GRZ.GRZ('')
print ('CNNs loaded Successfully')

httpd = HTTPServer(('127.0.0.1', PORT), SimpleHTTPRequestHandler)
#httpd = HTTPServer(('91.226.83.146', PORT), SimpleHTTPRequestHandler)
print("serving at port", PORT)
httpd.serve_forever()
