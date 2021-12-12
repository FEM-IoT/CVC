#!/bin/sh
python3 server_detection.py --model1="TensorFlow/femiot_crossing_yolov3.pb" --classNames1="TensorFlow/femiot.names" --model2="TensorFlow/yolov3.pb" --classNames2="TensorFlow/coco.names"

