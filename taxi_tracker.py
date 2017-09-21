import argparse
import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import urllib

yellow_low = np.array([20,100,100])
yellow_high = np.array([30,255,255])

GRAPH_LOCAL_PATH = 'data/taxi_reader_graph.pb'
GRAPH_URL_PATH = 'https://github.com/Vzzzz/taxi_tracker/raw/master/data/taxi_reader_graph.pb'

DETECTION_BOUND = 0.3

def ocvDetect(frame):
    """Returns frame with selecter yellow object
        Pipeline:
            1)blur
            2)rgb2hsv
            3)mask by color
            4)canny (edge detection)
            5)contours
            ...
            6)profit.
    """
    filtered_frame = cv2.medianBlur(frame, 11)
    hsv_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)
    masked_frame = cv2.inRange(hsv_frame, yellow_low, yellow_high)
    canny_frame = cv2.Canny(masked_frame, 50, 100)
    image, contours, h = cv2.findContours(canny_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try: hierarchy = h[0]
    except: hierarchy = []

    height, width = canny_frame.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    outframe = frame.copy()

    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        #This may (must?) be configurable
        if w > 50 and h > 50 and w < width/2 and h < height/2:
            cv2.rectangle(outframe, (x,y), (x+w,y+h), (255, 0, 0), 2)

    return outframe

def ocvColorTest(subframe):
    """Defines the peak color of image (subframe) and compares with yellow"""
    hsv = cv2.cvtColor(subframe, cv2.COLOR_BGR2HSV)
    histr = cv2.calcHist([hsv],[0],None,[256],[0,256])
    color_key = np.argmax(histr)
    if color_key >= yellow_low[0] and color_key <= yellow_high[0]:
        return True
    return False

def loadTfGraph():
    """Loads TensorFlow detection graph. If not found locally, downloads it from github"""
    print('Loading detection graph')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.GraphDef()
        if not os.path.isfile(GRAPH_LOCAL_PATH):
            print('Graph file not found, downloading')
            graphfile = urllib.FancyURLopener()
            graphfile.retrieve(GRAPH_URL_PATH, GRAPH_LOCAL_PATH)
            print('Graph saved in %s'%GRAPH_LOCAL_PATH)
        with tf.gfile.GFile(GRAPH_LOCAL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
            return detection_graph
    return None

def tfDetect(frame, detection_graph, colorTest=False):
    """This is a mostly copy-paste from Tensorflow Object Detection API example.
        It takes frame, detects objects with previously recorded graph and in case of
        color test checks what color of object is below the recognized rectangle
    """
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            frame_np_expanded = np.expand_dims(frame, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_np_expanded})
            outframe = frame.copy()
            _, height, width, _ = frame_np_expanded.shape
            for b,s in zip(boxes[0], scores[0]):
                if s > DETECTION_BOUND:
                    xmin = int(b[1] * width)
                    ymin = int(b[0] * height)
                    xmax = int(b[3] * width)
                    ymax = int(b[2] * height)
                    #This may be conigurable
                    if (xmax - xmin) > 0.4*width and (ymax-ymin) > 0.4*height:
                        continue
                    if colorTest==True:
                        crop = frame[ymin:ymax, xmin:xmax]
                        if not ocvColorTest(crop):
                            continue
                    cv2.rectangle(outframe, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            return outframe


def main(args):
    INPUT_PATH = args.input
    OUTPUT_PATH = args.output
    MODE = args.mode
    if INPUT_PATH is None:
        print("Please set the input file")
        exit()
    if OUTPUT_PATH is None:
        print("There is no output file defined")
        exit()
    if len(MODE) > 1 or MODE[0] not in 'otb':
        print("Incorrect mode selected")
        exit()
    print(INPUT_PATH, OUTPUT_PATH, MODE)
    if MODE == 't' or MODE == 'b':
        detection_graph = loadTfGraph()
    reader = cv2.VideoCapture(INPUT_PATH)
    frames_total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = reader.get(cv2.CAP_PROP_FPS)
    print(frames_total, width, height, fps)
    try:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
        skipped_frames = 0
        for i in range(frames_total):
            ret, frame = reader.read()
            if ret==True:
                if MODE=='o':
                    outframe = ocvDetect(frame)
                if MODE=='t':
                    outframe = tfDetect(frame, detection_graph, colorTest=False)
                if MODE=='b':
                    outframe = tfDetect(frame, detection_graph, colorTest=True)
                writer.write(outframe)
            else:
                skipped_frames += 1
            sys.stdout.write('\r')
            sys.stdout.write("[%-100s] %d%%" % ('='*int(100*(i+1)/frames_total), int(100*(i+1)/frames_total)))
            sys.stdout.flush()
        if skipped_frames > 0:
            print('\nSkipped frames: %d'%skipped_frames)
        print('Passed')
        reader.release()
        writer.release()
    except Exception:
        print('\nStopped')
        print(Exception.message)
        if reader is not None and reader.isOpened():
            reader.release()
        if writer is not None:
            writer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to input file', default=None)
    parser.add_argument('--output', help='path to output file', default=None)
    parser.add_argument('--mode', help='detection mode (o - opencv, t - tensorflow, b - both)', default='b')
    args = parser.parse_args()
    main(args)
