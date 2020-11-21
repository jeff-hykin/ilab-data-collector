from toolbox.globals import INFO, PATHS, PARAMETERS, FS
from toolbox.video_tools import Video
from toolbox.face_tools.expressions.Facial_expressions_detection import network_output as get_emotion_data
from toolbox.face_tools.expressions.Facial_expressions_detection import preprocess_face
import cv2 

# face_cascade = cv2.CascadeClassifier(PATHS['haarcascade_frontalface_default'])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def get_faces(image):
    face_dimensions = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    cropped_faces = [ image[y:y + h, x:x + w] for x, y, w, h in face_dimensions ]
    return cropped_faces, face_dimensions

# TODO: keep track of which videos are done

which_emotion = "happy"
for video_path in FS.list_files(PATHS["videoStorage"]):
    # 
    # Scan Video
    # 
    video = Video(video_path)
    emotion_strength_per_frame = []
    max_duration = 11 * 60 # seconds
    min_duration  =  5 * 60 # seconds
    if min_duration < video.duration() < max_duration:
        for each_frame in video.frames():
            emotion_value = 0
            cropped_faces, face_dimensions = get_faces(each_frame)
            print('len(cropped_faces) = ', len(cropped_faces))
            for each_face in cropped_faces:
                emotion_data = get_emotion_data(preprocess_face(each_face))
                # opt for the largest value in the frame
                if emotion_data["probabilities"][which_emotion] > emotion_value:
                    emotion_value = emotion_data["probabilities"]["happy"]
            emotion_strength_per_frame.append(emotion_value)
    if len(emotion_strength_per_frame) > 0:
        break
    
    
    # Use algorithm for finding moments
    
    