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

## Com executar l'Script de demo
Tal com s'ha dit anteriorment, el codi està preparat per ser executat al CVC Webservices. El CVC Webservices es un webservice del CVC que està basat en Flask i que utilitza computació GPU per executar els algorismes més actuals en Visió per Computador desenvolupats al Centre de Visió per Computador (CVC).
No obstant, també s'ha inclòs un script anomenat "test_detection.py" que permet executar el mateix codi sense l'arquitectura del CVC Webservices. Utilitzant el fitxer Dockerfile que s'inclou al repositori, es pot muntar un sistema per executar aquest script sense cap altre mòdul de software requerit pel servidor. És important remarcar que el sistema executant aquest codi ha de tenir instal·lada una targeta GPU (Graphics Processing Unit) per poder executar correctament el codi.

Una vegada la imatge del container s'ha creat utilitzant la comanada:

	     sudo docker build --tag pedestriandetection .
des de la carpeta on hi ha el Dockerfile, es necessari arrencar una sessió interactiva amb la imatge creada utilitzant la següent comanda:

        sudo nvidia-docker run -it -v $(pwd)/videos:/videos -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix pedestriandetection bash

Per executar l'script a dins el container, s'ha de cridar amb els següents paràmetres:

        python3 test_detection.py --model1 ./TensorFlow/femiot_crossing_yolov3.pb --classNames1 ./TensorFlow/femiot.names --model2 ./TensorFlow/yolov3.pb --classNames2 ./TensorFlow/coco.names ../videos/UAB-SAF_008_anonimized.avi

On ../videos/UAB-SAF_008_anonimized.avi es el nom del vídeo que es vol analitzar que està al directori que s'ha muntat a la sessió interactiva.
La sortida de l'script son les estadístiques calculades. L'script també pot mostrar una imatge amb informació sobre on son la/es zona/es del/s pas/sos de vianants i amb les trajectòries dels vianants i les bicicletes.
Si la imatge no pot ser mostrada, es possible que calgui executar la següent comanda
	xhost+local:docker
abans de crear la sessió interactiva amb el Docker.

## Fitxers i Directoris
El repositori conté bàsicament 5 carpetes:

 - Docker: Conté el fitxer de Dockerfile i tots els fitxers necessaris per crear un entorn on executar el codi.
 - Documents: Conté un fitxer que explica el context del projecte, com s'ha desenvolupat i que dóna algunes directrius per executar el codi.
 - Storage: Aquest directori es necessari per l'arquitectura del servidor CVC Webservices.
 - Tensorflow: Conté tots els fitxers necessaris per executar els models de Tensorflow inclosos al codi.
 - Test: Conté fitxers de vídeo per poder testejar fàcilment el codi.
 
 El repositori també conté 5 fitxers:
 - docker-entrypoint.sh: És utilitzat per l'arquitectura del servidor CVCWebservices.
 - docker-entrypoint-debug.sh: És utilitzat per l'arquitectura del servidor CVCWebservices.
 - femiotUtils.py: Conté totes les funcions utilitzades pels scripts principals per implementar la funcionalitat.
 - server_detection.py: Aquest es l'script utilitzat per l'arquitectura CVCWebservices per implementar la funcionalitat al CVCWebservices.
 - test_detection.py: Aquest es l'script que pot ser fàcilment utilitzat per testejar el codi en un entorn sense servidor.
