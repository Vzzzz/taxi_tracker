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
    histr = cv2.calcHist([hsv],['r'],None,[256],[0,256])
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
                    outframe = ocvColorTest(frame)
                if MODE=='b':
                    outframe = None
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
    except:
        print('\nStopped')
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
