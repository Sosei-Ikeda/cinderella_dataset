# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 20:38:24 2020

@author: Sosei Ikeda
"""

import glob
from PIL import Image
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import cv2
import numpy as np

KEY = 'fabbf6804296424baf139a881c3a659b'
ENDPOINT = 'https://sosei-resource.cognitiveservices.azure.com/'
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

def face_position(faceDictionary):
    rect = faceDictionary.face_rectangle
    x = rect.left
    y = rect.top
    h = rect.height
    w = rect.width
   
    l = h if h > w else w

    x = int(x - (0.1 * l))
    y = int(y - (0.1 * l))
    l = int(1.2 * l)
    
    return x,y,l

def crop_face(imageUrl,x,y,l):
    img = Image.open(imageUrl)
    imgArray = np.asarray(img)
    imgArrayCropped = imgArray[y:y+l, x:x+l]
    imgArrayCropped = cv2.cvtColor(imgArrayCropped, cv2.COLOR_BGR2RGB)
    
    return imgArrayCropped

image_url_array = [file for file in glob.glob('*.jpg')]

for i in range(175,len(image_url_array)):
    rawimg = cv2.imread(image_url_array[i])
    cv2.imwrite(f'rawimg/{i}.jpg', rawimg)
    detected_faces = face_client.face.detect_with_url(url="https://raw.githubusercontent.com/Sosei-Ikeda/cinderella_dataset/master/"+image_url_array[i], detection_model='detection_02')
    if not detected_faces:
        print(f'No face detected from this image : {image_url_array[i]}')
    else:
        for face in detected_faces:
            left, top, length = face_position(face)
            cropped_face = crop_face(image_url_array[i],left,top,length)
            cv2.imwrite(f'img/{i}.jpg', cropped_face)

#i = 141
#detected_faces = face_client.face.detect_with_url(url="https://raw.githubusercontent.com/Sosei-Ikeda/cinderella_face/master/"+image_url_array[i], detection_model='detection_02')
#if not detected_faces:
#    print(f'No face detected from this image : {image_url_array[i]}')
#else:
#    print(len(detected_faces))
#    left, top, length = face_position(detected_faces[0])
#    cropped_face = crop_face(image_url_array[i],left,top,length)
#    cv2.imwrite(f'img/{i}.jpg', cropped_face)