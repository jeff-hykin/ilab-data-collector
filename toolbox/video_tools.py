from subprocess import call
import cv2
# local imports
from toolbox.file_system_tools import FS


class Video(object):
    def __init__(self, path=None):
        self.path = path
    
    def frames(self):
        """
        returns: a generator, where each element is a image as a numpyarray 
        """
        if not FS.is_file(self.path):
            raise Exception(f"Error, tried opening {self.path} but I don't think that is a file")
        
        # detect type
        *folders, file_name, file_extension = FS.path_pieces(self.path)
        # video_format = None
        # print('file_extension = ', file_extension)
        # if file_extension == ".avi":
        #     video_format = cv2.VideoWriter_fourcc(*'aaaa')
        # print('video_format = ', video_format)
        
        # Path to video file 
        video_capture = cv2.VideoCapture(self.path)
        # Check if video opened successfully
        if (video_capture.isOpened()== False): 
            raise Exception(f"Error, tried opening {self.path} with cv2 but wasn't able to")
        
        # checks whether frames were extracted 
        success = 1
        while True: 
            # function extract frames 
            success, image = video_capture.read()
            if not success:
                video_capture.release()
                return None
            yield image
    
    def fps(self):
        video_capture = cv2.VideoCapture(self.path)
        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()
        return fps
    
    def frame_count(self):
        video_capture = cv2.VideoCapture(self.path)
        frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        video_capture.release()
        return frame_count
        
    def duration(self):
        """
        returns the duration of the video in seconds (float)
        """
        video_capture = cv2.VideoCapture(self.path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            raise Exception("Unable to get video duration.\nThe FPS of the video is used to calculate duration, however this video returned an FPS of 0.")
        total_number_of_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = float(total_number_of_frames) / float(fps)
        video_capture.release()
        return duration
        
    def frame_width(self):
        video_capture = cv2.VideoCapture(self.path)
        frame_width  = int(video_capture.get(3))
        video_capture.release()
        return frame_width
    
    def frame_height(self):
        video_capture = cv2.VideoCapture(self.path)
        frame_height = int(video_capture.get(4))
        video_capture.release()
        return frame_height
    
    def save_with_labels(self, list_of_labels, to=None):
        # 
        # extract video data
        # 
        video_capture = cv2.VideoCapture(self.path)
        frame_width  = int(video_capture.get(3))
        frame_height = int(video_capture.get(4))
        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()
        
        # 
        # create new video source
        # 
        frame_dimensions = (frame_width, frame_height)
        if to is None:
            *folders, name, ext = FS.path_pieces(self.path)
            output_file = FS.join(*folders, name+".labelled"+ext)
        else:
            output_file = to
        new_video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_dimensions)
        # Read until video is completed
        for each_frame, each_label in zip(self.frames(), list_of_labels):
            if each_frame is None:
                break
            # Write text to frame
            text = str(each_label)
            text_location = (100, 100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1
            color = (255, 255, 255)
            new_video.write(cv2.putText(each_frame, text, text_location, font, thickness, color, 2, cv2.LINE_AA))
        
        # combine the resulting frames into a video
        new_video.release()

    @classmethod
    def create_from_frames(self, list_of_frames, save_to=None, fps=30):
        # check
        if len(list_of_frames) == 0:
            raise Exception('The Video.create_from_frames was given an empty list (no frames)\nso a video cannot be made')
        elif save_to == None:
            raise Exception('The Video.create_from_frames was given no save_to= location so it doesn\'t know where to save the file')
        
        # 
        # create new video source
        # 
        frame_height, frame_width = list_of_frames[0].shape[:2]
        frame_dimensions = (frame_width, frame_height)
        new_video = cv2.VideoWriter(save_to, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_dimensions)
        # add all the frames
        for each_frame in list_of_frames:
            new_video.write(each_frame)    
        
        # combine the resulting frames into a video, which will write to a file
        new_video.release()

