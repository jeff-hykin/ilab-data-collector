import torch
import torch.nn as nn
import time
import numpy as np
import cv2 as cv
import cv2


# relative imports
from toolbox.globals import PATHS
from toolbox.face_tools.recognition.vgg import VGG

WITH_GPU = False
LABEL2EMOTION = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt", "none", "uncertain", "non-face"]
    
has_initilized = False
face_cascade = None
checkpoint = None
net = None
device = None

def _init():
    global has_initilized
    global face_cascade
    global net
    global device
    
    if has_initilized == False:
        # pull up the model network
        net = VGG('VGG19')
        
        # pick which hardware
        device = torch.device("cuda:0") if WITH_GPU else torch.device("cpu")
        
        checkpoint = torch.load(PATHS['test_model.t7'], map_location=device)
        net.load_state_dict(checkpoint['net'])
        if WITH_GPU:
            net = net.to(device)
        net.eval()

        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        has_initilized = True

def network_output(input_face):
    """
    @input_face should be a preprocessed torch.FloatTensor
    
    @@return {
        "most_likely" : (string from LABEL2EMOTION),
        "probabilities": {
            "neutral" : (float ≥0 ≤100),
            "happy" : (float ≥0 ≤100),
            (etc)
        }
    }
    """
    _init()
    output = {}
    logits = net(input_face)
    c = int(torch.argmax(logits, 1))
    output["most_likely"] = LABEL2EMOTION[c]
    prob = torch.nn.functional.softmax(logits[0], dim=0) * 100.0
    output["probabilities"] = {}
    for each_index, each_value in enumerate(prob):
        output["probabilities"][LABEL2EMOTION[each_index]] = float(each_value)
    return output

def preprocess_face(face_img):
    _init()
    input_face = cv.resize(face_img, (300, 300), cv.INTER_CUBIC)
    input_face = input_face.astype(np.float32)
    input_face = input_face / 255.0
    input_face = np.expand_dims(input_face, axis=0)
    input_face = np.transpose(input_face, (0, 3, 1, 2))
    input_face = torch.FloatTensor(input_face)
    if WITH_GPU:
        input_face = input_face.to(device)
    return input_face

def label_all(frame):
    """
    @frame: an image represented as a numpy array (standard opencv image)
    @@return: a list with each element being as follows 
        {
            "x" : x,
            "y" : y,
            "width" :  w,
            "height" :  h,
            "emotion_vgg19_0.0.2": {
                "most_likely" : (string from LABEL2EMOTION),
                "probabilities": {
                    "neutral" : (float ≥0 ≤1),
                    "happy" : (float ≥0 ≤1),
                    (etc)
                }
            }
        }
    """
    # make sure everything is avalible
    _init()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1, minSize=(100, 100), flags=cv.CASCADE_SCALE_IMAGE)
    output = []
    for each in faces:
        # pick the first face avalible
        (x, y, w, h) = each
        # clip out the face
        input_face = frame[y:y + h, x:x + w]
        output.append({
            "x" : x,
            "y" : y,
            "width" :  w,
            "height" :  h,
            "emotion_vgg19_0.0.2" : network_output(preprocess_face(input_face)),
        })
    return output
            

def inference(video_file):
    # make sure everything is avalible
    _init()
    
    font = cv.FONT_HERSHEY_SIMPLEX
    cap = cv.VideoCapture(video_file)

    now = time.time()
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1, minSize=(100, 100), flags=cv.CASCADE_SCALE_IMAGE)
        if len(faces) > 0:
            # pick the first face avalible
            (x, y, w, h) = faces[0]
            face = frame[y:y + h, x:x + w]
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            final_face = cv.resize(face, (300, 300), cv.INTER_CUBIC)

            input_face = final_face.astype(np.float32)
            input_face = input_face / 255.0
            input_face = np.expand_dims(input_face, axis=0)
            input_face = np.transpose(input_face, (0, 3, 1, 2))
            input_face = torch.FloatTensor(input_face)
            if WITH_GPU:
                input_face = input_face.to(device)
            logits = net(input_face)
            c = int(torch.argmax(logits, 1))
            prob = torch.nn.functional.softmax(logits[0], dim=0) * 100.0
            cv.putText(frame, LABEL2EMOTION[c], (100, 50), font, 2, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(frame, "Neutral %d" % prob[0], (20, 100), font, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(frame, "Happy %d" % prob[1], (20, 150), font, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(frame, "Sad %d" % prob[2], (20, 200), font, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(frame, "Surprise %d" % prob[3], (20, 250), font, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(frame, "Fear %d" % prob[4], (20, 300), font, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(frame, "Disgust %d" % prob[5], (20, 350), font, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(frame, "Anger %d" % prob[6], (20, 400), font, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(frame, "Contempt %d" % prob[7], (20, 450), font, 1, (255, 255, 255), 2, cv.LINE_AA)

            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()


#
# example?
#
if __name__ == '__main__':
    image = cv.imread("./")