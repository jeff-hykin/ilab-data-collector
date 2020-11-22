import traceback 
import json
from toolbox.globals import INFO, PATHS, PARAMETERS, FS
from toolbox.video_tools import Video
from toolbox.face_tools.expressions.Facial_expressions_detection import network_output as get_emotion_data
from toolbox.face_tools.expressions.Facial_expressions_detection import preprocess_face
from source.moment_selection.moment_selection_algorithm import logarithmic_cluster
import cv2 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def get_faces(image):
    face_dimensions = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    cropped_faces = [ image[y:y + h, x:x + w] for x, y, w, h in face_dimensions ]
    return cropped_faces, face_dimensions

# TODO: keep track of which videos are done

moments = []
which_emotion = "happy"
done_videos = set()
# busy wait for more videos
while True:
    for video_path in FS.list_files(PATHS["videoStorage"]):
        try:
            # don't repeat videos
            if video_path in done_videos:
                continue
            else:
                done_videos.add(video_path)

            *folders, filename, ext = FS.path_pieces(video_path)
            # 
            # Scan Video
            # 
            video = Video(video_path)
            print('filename = ', filename)
            emotion_strength_per_frame = []
            max_duration = 11 * 60 # seconds
            min_duration = 5 * 60 # seconds
            duration     = video.duration()
            if min_duration < duration < max_duration:
                print('duration = ', duration)
                frame_count = 0
                for each_frame in video.frames():
                    print('frame_count = ', frame_count)
                    frame_count += 1
                    emotion_value = 0
                    cropped_faces, face_dimensions = get_faces(each_frame)
                    for each_face in cropped_faces:
                        emotion_data = get_emotion_data(preprocess_face(each_face))
                        # opt for the largest value in the frame
                        if emotion_data["probabilities"][which_emotion] > emotion_value:
                            emotion_value = emotion_data["probabilities"]["happy"]
                    emotion_strength_per_frame.append(emotion_value/100.0)
                    
                duration_of_single_frame = (duration+0.0) / frame_count
                print('emotion_strength_per_frame = ', emotion_strength_per_frame[:10])
                clusters_of_frames, confidences = logarithmic_cluster(emotion_strength_per_frame)
                time_segments = [
                    (
                        each_start*duration_of_single_frame,
                        each_end*duration_of_single_frame,
                    ) for each_start, each_end in clusters_of_frames
                ]
                print('time_segments = ', time_segments)
                for each_time_segment, confidence in zip(time_segments, confidences):
                    moments.append({
                        "type": "segment",
                        "videoId": filename,
                        "observer": "haarcascade-vgg16-v1",
                        "isHuman": False,
                        "confirmedBySomeone": False,
                        "rejectedBySomeone": False,
                        "observation": {
                            "label": "happy",
                            "labelConfidence": confidence,
                        },
                        "startTime": each_time_segment[0],
                        "endTime": each_time_segment[1],
                    })
                print("saving")
                FS.write(json.dumps(moments), to=FS.join(FS.dirname(__file__),"moments.json"))
                
            else:
                print("video skipped")
        except Exception as error:
            print('error = ', error)
            traceback.print_exc() 
            
        
    