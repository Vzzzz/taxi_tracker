Taxi tracker 0.1a

This is a simple Python 2.7 script that may be used to detect yellow taxis on video.
This is NOT an online solution unless source FPS is about 3.5

Usage:
python taxi_tracker.py --input in.mp4 --output out.avi --mode b

where
* input sets path to input file
* output to output
* mode [o, t, b (by default)] choses a detection mode: o - opencv (select by color), t - tensorflow network with pretrained graph, b - is like t, but then it checks if selected car is yellow

Script uses OpenCV for video and imagig routines, it loads video, analyze each frame (there is no region tracking or anything like this), detects objects with two algorithms and stores frames into another videofile (coded with XVID)

Dependencies:
* opencv-python
* tensorflow
* numpy
* urllib (in case if it needs to download pretrained graph)
