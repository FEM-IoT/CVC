# CVC - P2 - Pedestrian and Bicycles Detection

This repository contains the code to implement the funcionality of the FEMIOT P2 project called "Valorization of data from de IoT". In concrete, this repository contains the code for the use case "Package algorithm for the detection of pedestrians and light weight vehicles".
This repository contains all the code and files to process the video in the CVC Webservices platform. It also contains a script to execute the code out of the server architecture, to easily check the functionality in a Linux platform provided with a GPU (Graphics Processing Unit) hardware.
The repository also contains a Dockerfile to easily setup the environment for running the code.

## Summary
The contribution of the CVC (Computer Vision Center) to the "Valorization of the data from the IoT" project is a module that recieves a video from a Zebra Crossing Area and analyzes the behaviour of the people and light vehicles crossing it. The aim of this project is to be able to determine whether a certain Zebra Crossing Area is a special problematic one, or if it has the traffic line periods of time correctly setup.

The module receives a video and first returns some information about the received video:
 - Duration.
 - Frames per second (fps).
 - Number of Zebra Crossing Areas.
 - Width of the video.
 - Height of the video.

 and secondly, for each Zebra Crossing Area in the video, the following statistics:

 - Average time used to cross in direction A / B.
 - Average time used to wait before crossing in direction A / B.
 - Maximum time used to cross in direction A / B.
 - Maximum time used to wait before crossing in direction A / B.
 - Minimum time used to cross in direction A / B.
 - Minimum time used to wait before crossing in direction A / B.
 - Number of Bicycles crossing in direction A / B.
 - Number of Persons crossing in direction A / B.

The technical approach used to perform this analysis is based on Deep Learnning.
The system first runs a detector (based on a Yolo Detector) to detect the Zebra Crossing Areas in all the video. Once the Zebra Crossing Areas are clear, it runs a detector (also based on Yolo) to detect pedestrains and bicycles. With all the detections and the tracking of them, it determines how many frames each person and bicycle is crossing or waiting, and with this information it calculates the time used for each individual. The detections are also used to determine the waiting areas at the borders of the crossing areas.

## Running Demo Script
As it has been said, the code is setup to be run in the CVC Webservices server. The CVC Webservices is a CVC Webservice that it is based on Flask and that it uses GPU computation to run the cutting edge algorithms in Computer Vision developed by the Computer Vision Center (CVC).
However, a script called "test_detection.py " is provided to be able to run the same code without the CVC Webservices architecture. Using the Dockerfile provided, a system can be setup to run this script without any the other software modules required by the server. It is important to remark that the system running this code has to have a GPU (Graphics Processing Unit) hardware to run.

Once the image of the container is created with docker using the command:

	     sudo docker build --tag pedestriandetection .
from the folder where the Dockerfile is, it is required to start an interactive session with the created image with the following command:

        sudo nvidia-docker run -it -v $(pwd)/videos:/videos -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix pedestriandetection bash

To run the script into the container it has to be called with the following parameters:

        python3 test_detection.py --model1 ./TensorFlow/femiot_crossing_yolov3.pb --classNames1 ./TensorFlow/femiot.names --model2 ./TensorFlow/yolov3.pb --classNames2 ./TensorFlow/coco.names ./Test/video.mp4` 

Where ./Test/video.mp4 is the name of the video to be run.

The output of the script are the statistics calculated. The script can also show an image with information about where the Zebra Crossing areas are, and with the trajectories of the pedestrians and bicycles.
If the image cannot be output, it is possible that it is required to call
`xhost + loca: docker
before the interactive session with the Docker container is started.

## Files and Directories
The repository contains 5 basic folders:

 - Docker: Contains the Dockerfile and all the needed files to setup the environment to run the code.
 - Documents: Contains an explanation of how the project has been devolped and how to easily run the code.
 - Storage: This folder is required for the CVC Webservices server architecture.
 - Tensorflow: Contains all the required files for the Tensorflow models used in the code.
 - Test: Contains video files to easily test the code.
 
 The repository also contains 5 files:
 - docker-entrypoint.sh: It is used by the CVC Webservices server architecture.
 - docker-entrypoint-debug.sh: It is used by the CVC Webservices server architecture.
 - femiotUtils.py: It contains all the functions used by the main scripts to implement the functionality
 - server_detection.py: This is the script run by the CVC Webservices architecture to implement the functionality in the CVC Webservices.
 - test_detection.py: This is a script that can be easily used to test the code in a non-server environment.
