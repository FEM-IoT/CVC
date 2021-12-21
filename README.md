# CVC - P2 - Detecció de Vianants i Bicicletes
Aquest repositori conté el codi per implementar la funcionalitat del projecte FEM-IoT P2 anomenat "Valorització de les dades de la IoT". Concretament, aquest reposotori conté el codi del cas d'ús "Algorisme d'empaquetat per a la detecció de vianants i vehicles lleugers".
En aquest reposotori hi ha tot el codi i fitxers per processsar un video a la plataforma CVC Webservices. A més, conté un script per executar el codi sense tota l'arquitectura del servidor, per fàcilment comprovar la funcionalitat en una plataforma Linux proveïda d'una targeta GPU (Craphic Processing Unit).
El repositori també conté un fitxer Dockerfile per fàcilment poder muntar un entorn on executar el codi.

## Resum
La contribució del CVC (Centre de Visió per Computador) al projecte "Valorització de les dades de la IoT" es un mòdul que rep un vídeo d'una zona on hi ha un o diversos passos de vianants, i que analitza el comportament de la gent i els vehicles lleugers que el creuen. L'objectiu d'aquest projecte és ser capaços de determinar si un pas de vianants es especialment problemàtic, o si els períodes de temps assignats a cada etapa semafòrica tenen els valors més adients.

El mòdul rep un vídeo i primer de tot retorna informació del vídeo rebut:
  - Durada
  - Número de imatges (frames) per segon (fps).
  - Número de passos de vianants.
  - Amplada del vídeo.
  - Alçada del vídeo.
  
a continuació, per cadascun dels passos de vianants del vídeo, retorna les següents estadístiques:

 - Temps mitjà utilitzat per creuar en direacció A / B.
 - Temps mitjà utilitzat esperant abans de creuar en direcció A / B.
 - Temps màxim utilitzat per creuar en direacció A / B.
 - Temps màxim utilitzat esperant per creuar en direcció A / B.
 - Temps mínim utilitzat per creuar en direcció A / B.
 - Temps mínim utilitzat esperant per creuar en direcció A / B.
 - Número de Bicicletes creuant en direcció A / B.
 - Número de Persones creuant en direcció A / B.

L'enfocament tècnic utilitzat per realitzar aquest anàlisis està basat en tècniques de Deep Learnning.
El sistema en primer lloc executa un detector (basat en un detector Yolo) per detectar les zones del pas de vianants a tot el vídeo. Una vegada les zones dels passos de vianants estàn clares, executa un detector (també basat en Yolo) per detectar els vianants i les bicicletes. Amb totes les deteccions, i amb el tracking d'aquestes deteccions, determina durant quants frames cada persona o bicicleta estàn creuant o esperant, i amb aquesta informació calcula el temps utilitzat per cada individu. Les deteccions també son utilitzades per determinar les zones d'espera als límits de les zones de passos de vianants.

## Running Demo Script
As it has been said, the code is setup to be run in the CVC Webservices server. The CVC Webservices is a CVC Webservice that it is based on Flask and that it uses GPU computation to run the cutting edge algorithms in Computer Vision developed by the Computer Vision Center (CVC).
However, a script called "test_detection.py " is provided to be able to run the same code without the CVC Webservices architecture. Using the Dockerfile provided, a system can be setup to run this script without any the other software modules required by the server. It is important to remark that the system running this code has to have a GPU (Graphics Processing Unit) hardware to run.

Once the image of the container is created with docker using the command:

	     sudo docker build --tag pedestriandetection .
from the folder where the Dockerfile is, it is required to start an interactive session with the created image with the following command:

        sudo nvidia-docker run -it -v $(pwd)/videos:/videos -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix pedestriandetection bash

To run the script into the container it has to be called with the following parameters:

        python3 test_detection.py --model1 ./TensorFlow/femiot_crossing_yolov3.pb --classNames1 ./TensorFlow/femiot.names --model2 ./TensorFlow/yolov3.pb --classNames2 ./TensorFlow/coco.names ./Test/video.mp4

Where ./Test/video.mp4 is the name of the video to be run.

The output of the script are the statistics calculated. The script can also show an image with information about where the Zebra Crossing areas are, and with the trajectories of the pedestrians and bicycles.
If the image cannot be output, it is possible that it is required to call
`xhost + loca: docker
before the interactive session with the Docker container is started.

## Files and Directories
The repository contains 4 basic folders:

 - Docker: Contains the Dockerfile and all the needed files to setup the environment to run the code.
 - Storage: This folder is required for the CVC Webservices server architecture.
 - Tensorflow: Contains all the required files for the Tensorflow models used in the code.
 - Test: Contains video files to easily test the code.
 
 The repository also contains 5 files:
 - docker-entrypoint.sh: It is used by the CVC Webservices server architecture.
 - docker-entrypoint-debug.sh: It is used by the CVC Webservices server architecture.
 - femiotUtils.py: It contains all the functions used by the main scripts to implement the functionality
 - server_detection.py: This is the script run by the CVC Webservices architecture to implement the functionality in the CVC Webservices.
 - test_detection.py: This is a script that can be easily used to test the code in a non-server environment.
