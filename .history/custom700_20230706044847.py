import argparse
import ast
import importlib.util
# 448 packages
import os
import sys
import time
from threading import Thread

import cv2
import numpy as np

from centroidtracker import CentroidTracker


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture('rtsp://192.168.100.45:554/1/')
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
objects ={}
old_objects={}

# compare the co-ordinates for dictionaries of interest
def DictDiff(dict1, dict2):
   dict3 = {**dict1}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
               dict3[key] = np.subtract(dict2[key], dict1[key])
   return dict3

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

# Newly added co-ord stuff
obsFrames = 0
car_counts = {}
new_centroid1 = 0
new_centroid2 = 0
total_in = 0
total_out = 0
car_list1 = []
car_list2 = []
previous_centroids = {}
objectID_count = {}
difference = 0
#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # On the next loop set the value of these objects as old for comparison
    old_objects.update(objects)

    # Grab frame from video stream
    frame1 = videostream.read()
    
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    #rects variable
    rects =[]
    centroid0 = 0
    objectID = 0
    new_centroid1 = 0
    new_centroid2 = 0
    difference = 0
    ymin = 0
    xmin = 0
    ymax = 0
    xmax = 0
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            if object_name == 'car' or object_name == 'truck' or object_name == 'bus':
            
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                box = np.array([xmin,ymin,xmax,ymax])
            
                rects.append(box.astype("int"))
                
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (10, 255, 0), 2)
            
                # Draw label
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    #update the centroid for the objects
    objects = ct.update(rects)
    # calculate the difference between this and the previous frame
    x = DictDiff(objects,old_objects)
    
	# !loop over the tracked objects
    for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        centroid0 = int(centroid[0])
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    #count number of frames for direction calculation
    # obsFrames = obsFrames + 1  
    # if obsFrames % 1 == 0:
    # room1
    if ymax <= 1000 and xmax < 1380 and xmin > 570:
      print(objectID, xmin, ymin, xmax, ymax, centroid0, time.ctime(), flush=True)
      new_centroid1 = (xmin + xmax)//2
      # check counters1 for objectID
    if 600 < new_centroid1 < 1200:  
      if objectID not in previous_centroids:
        previous_centroids[objectID] = new_centroid1
      else:
        if objectID in previous_centroids:
          difference = new_centroid1 - previous_centroids[objectID]
          # previous_centroids[objectID] = centroids  # most recent centroids update
          if difference > 0:
            state = 'IN'
          else:
            state = 'OUT'
          # delet entries with the same objectID
          if objectID in objectID_count:
            del  objectID_count[objectID]
          if objectID in objectID_count: 
              if state in objectID_count[objectID]:
                  objectID_count[objectID][state] += 1
              else:
                  objectID_count[objectID][state] = 1
          else:
              objectID_count[objectID] = {'IN': 0, 'OUT': 0}
              objectID_count[objectID][state] = 1
          print(objectID_count, flush=True)    
          total_in = 0
          total_out = 0
          for objectID in objectID_count:
              total_in += objectID_count[objectID]["IN"]
              total_out += objectID_count[objectID]["OUT"]    

    #  room2
    if ymax > 1000 and xmax < 1380:
      print(objectID, xmin, ymin, xmax, ymax, centroid0, time.ctime(), flush=True)
        # # if ymax > 1000 and xmax < 1000:
      new_centroid2 = (xmin + xmax)//2
      # カウンター変数の初期化
      # check counters1 for objectID
    if 200 < new_centroid2 < 1200:
      if objectID not in previous_centroids:
        previous_centroids[objectID] = new_centroid2
      else:
        if objectID in previous_centroids:
          defference = new_centroid2 - previous_centroids[objectID]
          if difference > 0:
            state = 'IN'
          else:
            state = 'OUT'
          if objectID in objectID_count:
            del  objectID_count[objectID]
          if objectID in objectID_count:
              if state in objectID_count[objectID]:
                  objectID_count[objectID][state] += 1
              else:
                  objectID_count[objectID][state] = 1
          else:
              objectID_count[objectID] = {'IN': 0, 'OUT': 0}
              objectID_count[objectID][state] = 1
          print(objectID_count, flush=True)    
          total_in = 0
          total_out = 0
          for objectID in objectID_count:
              total_in += objectID_count[objectID]["IN"]
              total_out += objectID_count[objectID]["OUT"]
            
        
    # 矩形黒の塗りつぶしボックスを作成
    cv2.rectangle(frame, (1700, 100), (2400, 700), (0, 0, 0), -1)
    # in_countとout_countをputTextにて表示
    cv2.putText(frame, f'Total IN : {total_in}', (1750, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Total OUT : {total_out}', (1750, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'STOP : {50+total_out-total_in}', (1750, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                  
      # Draw framerate in corner of frame
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (1700, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    #frame2 = cv2.resize(frame, (640, 360))
    frame2 = cv2.resize(frame, (960, 540))
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('GO2 car detector', frame2)
              

    # Press 'q' to quit and give the total tally
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
