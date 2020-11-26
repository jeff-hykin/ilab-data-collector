import traceback 
import json
from toolbox.globals import INFO, PATHS, PARAMETERS, FSL
from toolbox.video_tools import Video
from toolbox.face_tools.expressions.Facial_expressions_detection import network_output as get_emotion_data
from toolbox.face_tools.expressions.Facial_expressions_detection import preprocess_face
from source.moment_selection.moment_selection_algorithm import logarithmic_cluster
import cv2 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def get_faces(image):
    face_dimensions = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    cropped_faces = [ image[ y : y+h , x : x+w ] for x, y, w, h in face_dimensions ]
    return cropped_faces, face_dimensions

observer = "haarcascade-vgg16-v2"
observation_entries = []
done_videos = set(json.loads(FSL.read("./memory.json")))
# busy wait for more videos
while True:
    print("getting video list files")
    for video_path in FSL.list_files(PATHS["videoStorage"]):
        try:
            # don't repeat videos
            if video_path in done_videos:
                continue

            *folders, filename, ext = FSL.path_pieces(video_path)
            # 
            # Scan Video
            # 
            print('filename = ', filename)
            video = Video(video_path)
            emotion_strengths_per_frame = {
                "neutral": [],
                "happy": [],
                "sad": [],
                "surprise": [],
                "fear": [],
                "disgust": [],
                "anger": [],
                "contempt": [],
            }
            # try to get the duration, but if not, the delete the video file
            # cause it is probably corrupt
            try:
                duration = video.duration()
            except:
                print(f"couldn't get duration for {filename}, probably corrupt, deleting {filename}")
                FSL.delete(video_path)
                
            max_duration = 11 * 60 # seconds
            min_duration = 5 * 60 # seconds
            if min_duration < duration < max_duration:
                print('duration = ', duration)
                frame_count = 0
                for each_frame in video.frames():
                    if frame_count % 200 == 0:
                        print('frame_count = ', frame_count)
                    frame_count += 1
                    
                    cropped_faces, face_dimensions = get_faces(each_frame)
                    emotion_values = {
                        "neutral": 0,
                        "disgust": 0,
                        "surprise": 0,
                        "contempt": 0,
                        "anger": 0,
                        "happy": 0,
                        "sad": 0,
                        "fear": 0,
                    }
                    for each_face in cropped_faces:
                        emotion_data = get_emotion_data(preprocess_face(each_face))
                        for each_emotion_name in emotion_values:
                            if emotion_data["probabilities"][each_emotion_name] > emotion_values[each_emotion_name]:
                                emotion_values[each_emotion_name] = emotion_data["probabilities"][each_emotion_name]
                    for each_emotion_name in emotion_strengths_per_frame.keys():
                        emotion_strengths_per_frame[each_emotion_name].append(emotion_values[each_emotion_name]/100.0)
                
                duration_of_single_frame = (duration+0.0) / frame_count
                def compute_emotions(label_name, emotion_strength_per_frame):
                    print('emotion_strength_per_frame[:10] = ', emotion_strength_per_frame[:10])
                    clusters  = logarithmic_cluster(emotion_strength_per_frame)
                    # convert frame-indicies into timestamps
                    time_segments = [
                        (
                            each_start*duration_of_single_frame,
                            each_end*duration_of_single_frame,
                            each_confidence,
                        ) for each_start, each_end, each_confidence in clusters
                    ]
                    print('time_segments = ', time_segments)
                    for each_start, each_end, each_confidence in clusters:
                        # because the threshold is 0.5, the confidence will always be
                        # above 0.5, this normalized confidence just puts the remaining +0.5
                        # on a 0-1 scale
                        normalized_confidence = (each_confidence - 0.4999999999999) * 2
                        observation_entries.append({
                            "type": "segment",
                            "videoId": filename,
                            "observer": observer,
                            "isHuman": False,
                            "confirmedBySomeone": False,
                            "rejectedBySomeone": False,
                            "observation": {
                                "label": label_name,
                                "labelConfidence": normalized_confidence,
                            },
                            "startTime": each_start,
                            "endTime": each_end,
                        })
                    # if no clusters were found
                    if len(clusters) == 0:
                        average_non_confidence = sum(emotion_strength_per_frame)/len(emotion_strength_per_frame)
                        # change the 0 to 1 scale to a -1 to 1 scale
                        # this value should always end up being negative
                        # otherwise it would've been on-average above the threshold
                        normalized_video_confidence = (average_non_confidence*2)-1
                        observation_entries.append({
                            "type": "video",
                            "videoId": filename,
                            "observer": observer,
                            "isHuman": False,
                            "confirmedBySomeone": False,
                            "rejectedBySomeone": False,
                            "observation": {
                                "label": duration_of_single_frame,
                                "labelConfidence": normalized_video_confidence,
                            },
                        })
                # do it for each emotion
                for each_name, each_emotion_strength_per_frame in emotion_strengths_per_frame.items():
                    compute_emotions(each_name.capitalize(), each_emotion_strength_per_frame)
                
                print("saving")
                done_videos.add(video_path)
                FSL.write(json.dumps(observation_entries), to="observation_entries.json")
                FSL.write(json.dumps(list(done_videos)),   to="memory.json")
            else:
                print("video skipped")
            
            # remove the video after it has been processed
            FSL.delete(video_path)
        except Exception as error:
            print('error = ', error)
            traceback.print_exc() 
            
        
    