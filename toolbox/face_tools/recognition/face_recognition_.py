import numpy as np
import scipy as sp
import cv2
import os
import glob
import pickle
import scipy
import time
from time import sleep

# relative imports
import toolbox.face_tools.recognition.gen_feature as genfeat
import toolbox.face_tools.recognition.mtcnn as mtcnn
from toolbox.globals import PATHS, FS

feature_file = np.load(PATHS["feature_file.npz"])
UNKNOWN_LABEL = "Unknown"
THRESHOLD = 0.5
RUN_EXAMPLE = True
X11_AVALIBLE = True

# 
# create the feature array
# 
label_array = []
feature_array = []
for key in feature_file:
    label_array.append(key)
    feature_array.append(feature_file[key])
feature_array = np.array(feature_array)

def draw_label(image, point, label, emotion, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), -1)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
    cv2.putText(image, emotion, (point[0], point[1] + 20), font, font_scale, (255, 255, 255), thickness)

def get_label(feature):
    distance = float("inf")
    dist = scipy.spatial.distance.cdist(feature.reshape((1, feature.size)), feature_array, 'cosine')
    closest_index = np.argmin(dist)
    distance, label = dist[0][closest_index], label_array[closest_index] 

    return label if distance < THRESHOLD else UNKNOWN_LABEL 


def get_margins(face_margin, margin=1):
    (x, y, w, h) = face_margin[0], face_margin[1], face_margin[2] - face_margin[0], face_margin[3] - face_margin[1]
    margin = int(min(w, h) * margin / 100)
    x_a = int(x - margin)
    y_a = int(y - margin)
    x_b = int(x + w + margin)
    y_b = int(y + h + margin)
    return (x_a, y_a, x_b - x_a, y_b - y_a)


def face_recon(video_file):
    video_capture = cv2.VideoCapture(video_file)
    while video_capture.isOpened():

        # TODO: I'm not sure why this busywait is here or what it does
        if not video_capture.isOpened():
            sleep(5)
        
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if ret != True:
            break
        
        
        original_frame = frame.copy()
        _, boundingboxes, features, emotion = mtcnn.process_image(frame)
        
        # placeholder for cropped faces
        for shape_index in range(boundingboxes.shape[0]):
            (x, y, w, h) = get_margins(boundingboxes[shape_index, 0:4])
            
            if shape_index < len(features):
                label = get_label(features[shape_index])
            else:
                label = UNKNOWN_LABEL
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            draw_label(frame, (x, y), label, emotion)
        
        if not X11_AVALIBLE:
            print(f"label is: {label}")
        else:
            cv2.imshow('Face Detection', frame)
            
            # wait until user presses ESC key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # When everything is done, release the capture
    video_capture.release()
    return label, original_frame