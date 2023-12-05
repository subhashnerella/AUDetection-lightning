#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from detector import detect_faces
import numpy as np
from PIL import Image
from PIL.ImageDraw import Draw
import json
from tqdm import tqdm
import subprocess

# In[14]:

# Degrees of rotation in counter clockwise direction applied to image before running face detection.
# Example:
# original: 	^ 
# 90 degree: 	< 
# 180 degree:	v
# 270 degree:	> 

rotation = 90

# Thus, if image in 'Raw' directory looks like -> rotate by 90 to get upright image
# Similarly if image is like <- rotate by 270 to get upright image

# In[15]:
Patient = "Patient_109"
camera = "camera 2"
single_run = True # convert only a single video for testing rotation / script output
source_dir = "/data/datasets/ICU_Data/Sensor_Data/Patient_109/Video/P109 - working/P109 camera 2/"
dest_dir = "/data/datasets/ICU_Data/Sensor_Data/Patient_109/Face/Images/"


videos = os.listdir(source_dir)
if single_run:
    print("Processing single video from", source_dir)
    vid_list = [videos[0]]
else:
    print("Processing", len(videos), "videos from", source_dir)
    vid_list = videos
print("Rotating images by", rotation, "degrees")


for video in tqdm(vid_list, desc="videos"):
    source_vid = source_dir+video
    vid = video.split(".")[0]
    vid_dir_name = ('_').join([Patient, camera, vid]).replace(' ','-')
    raw_dir = dest_dir + "Raw/" + vid_dir_name + "/"
    out_dir = dest_dir + "To_be_annotated/" + vid_dir_name + "_output/"
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
        dest_imgs = (raw_dir + vid + "-%08d.jpg").replace(' ','-')
        #args = ['ffmpeg', '-loglevel', 'fatal', '-y', '-sseof', '-300', '-i', source_vid, '-qscale:v', '1', '-r', '1', dest_imgs]        
        args = ['ffmpeg', '-y', '-sseof', '-300', '-i', source_vid, '-qscale:v', '1', '-r', '1', dest_imgs]
        print((' ').join(args))
        subprocess.call(args)
        images = os.listdir(raw_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        face_bounding_boxes = {}

        face_bounding_boxes = {}
        for img in images:
            image = Image.open(raw_dir+'/'+img)
            image = image.rotate(rotation)
            try:
                bounding_boxes, landmarks = detect_faces(image)
            except:
                tqdm.write('no face found in '+img)
                continue
            if len(bounding_boxes) > 0:
                bbox = list(bounding_boxes[0][0:4])
                face_bounding_boxes[img] = bbox
                cropped = image.crop(bbox)
                cropped.save(out_dir+'/'+img, 'JPEG')
            else:
                face_bounding_boxes[img] = []

        with open(out_dir+'/'+vid+'.json', 'w') as f:
            json.dump(face_bounding_boxes, f)


