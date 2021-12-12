#!/bin/sh
python3 -m ptvsd --host 0.0.0.0 --port 5678 --wait server_detection.py --model1="TensorFlow/femiot_crossing_yolov3.pb" --classNames1="TensorFlow/femiot.names" --model2="TensorFlow/yolov3.pb" --classNames2="TensorFlow/coco.names"

