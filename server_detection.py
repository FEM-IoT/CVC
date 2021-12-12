# ========================================================================
# SERVER_DETECTION
# ------------------------------------------------------------------------
#
# Servidor que procesa una cola de imagenes para realizar Lecturas de Contador.
#
# @author	Technical Support Unit <ust@cvc.uab.es>
#
#           Centro de Vision por Computador
#           Edifici O - Universitat Autonoma de Barcelona.
#           08193 Cerdanyola del Valles, Barcelona, (SPAIN).
#           Tel. +(34) 93.581.18.28
#           Fax. +(34) 93.581.16.70
#
# @version  1.0.0 (9 de Noviembre de 2020)
#           2.0.0 (30 de Noviembre de 2021)
#
# ========================================================================
import numpy as np
import tensorflow as tf
import core.utils as utils
import argparse
import sys
import os
import cv2
import json
import redis
import time
import copy
import femiotUtils as ft
from norfair import Detection, Tracker, Video, draw_tracked_objects, draw_points	
from PIL import Image


# ========================================================================
MAX_WORKERS = 8


# ========================================================================
JSON_TAG_RESULTS = "results"
JSON_TAG_STATUS = "status"


# ========================================================================
QUEUE_HOST = "cvcws_message_queue"
QUEUE_PORT = 6379
QUEUE_DB = 0
QUEUE_ID = "cvcws_ust_pedestrianprediction_queue"

QUEUE_TAG_ID = "id"
QUEUE_TAG_VIDEO = "video"

SERVER_SLEEP = 0.025



# ========================================================================
# MAIN
# ------------------------------------------------------------------------
#
# Funcion principal del Servidor.
#
# ========================================================================
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", help="Model definition file (tflite) for Zebra Pass detection")
    parser.add_argument("--classNames1", help="Class Names file for Zebra Pass detection")
    parser.add_argument("--model2", help="Model definition file (tflite) for Pedestrians and bycicles")
    parser.add_argument("--classNames2", help="Class Names file for Pedestrians and bycicles")
    args = parser.parse_args()


    # to use 10% of GPU's memory (NOTE: Avoiding whole GPU memory allocation!!!)
    memory = 0.10
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = memory
 
    # First Graph (Detect zebra-crossings)
    pb_fileZebraPass = args.model1
    graphZebraPass = tf.Graph()
    return_tensorsZebraPass = ft.initTensors(pb_fileZebraPass, graphZebraPass)
    num_classesZebraPass = 1
    classesZebraPass = utils.read_class_names(args.classNames1)
    score_threshZebraPass = 0.4
    sess1 = tf.Session(graph=graphZebraPass)


    # Second Graph (Pedestrian Detect pedestrians)
    pb_filePedestrians = args.model2
    graphPedestrians = tf.Graph()
    return_tensorsPedestrians = ft.initTensors(pb_filePedestrians, graphPedestrians)
    num_classesPedestrians = 80
    classesPedestrians = utils.read_class_names(args.classNames2)
    score_threshPedestrians = 0.4
    sess2 = tf.Session(graph=graphPedestrians)


    # connect to message queue
    db_queue = redis.StrictRedis(host=QUEUE_HOST, port=QUEUE_PORT, db=QUEUE_DB)
    
    while True:
        # wait for message queue
        message = db_queue.blpop(QUEUE_ID, 0)[1]
        message_data = json.loads(message)

        # return processing response
        output = { JSON_TAG_RESULTS : [] }
        output[JSON_TAG_RESULTS].append({JSON_TAG_STATUS : "processing"})
        message = json.dumps(output)
        db_queue.set(message_data[QUEUE_TAG_ID], message)


        # start video processing
        try:
            # STEP 1 -----------------------------------------------------
            zebraPassesDetected = []
            vid = cv2.VideoCapture(message_data[QUEUE_TAG_VIDEO])
            videoInformation = ft.getVideoInformation(vid)
            [fps, width, height, frameCount] = videoInformation

            analyze_secs = 10
            analyze_count = min(analyze_secs * round(fps), frameCount)
            for x in range(analyze_count):
                return_value, frame = vid.read()
                if x % round(fps):
                    continue
                if return_value:
                    image = Image.fromarray(frame)
                    bboxes = ft.inferenceTensors(sess1, return_tensorsZebraPass, num_classesZebraPass, score_threshZebraPass, frame) 
                    zebraPassesDetected.append(bboxes)
                    print("Num Zebra passes is: ", len(bboxes))

            #Let's get the Zebra Pass Global Coordinates.
            zebraPassCoordinates = ft.getZebraPassAvgCoordinates(zebraPassesDetected)
            print("STEP 1: Done!")


            # STEP 2 -----------------------------------------------------
            vid = cv2.VideoCapture(message_data[QUEUE_TAG_VIDEO])
            #objecte tracker
            tracker = Tracker(distance_function=ft.euclidean_distance, distance_threshold=120, initialization_delay = 2)
            trajectoryPoints = []
            frameIndex=0
            detectionsWithClassAndwidth = [] #guardem la classe i l'amplada fora de l'objecte tracker per fer-les servir més endavant.

            while True:
                return_value, frame = vid.read()
                if return_value:
                    #Let's do the inference
                    prev_time = time.time()
                    image = Image.fromarray(frame)
                    if (frameIndex==0):
                        frame1 = copy.deepcopy(frame)
                        imageToShowTrajectories = Image.fromarray(frame1)

                    bboxes = ft.inferenceTensors(sess2, return_tensorsPedestrians, num_classesPedestrians, score_threshPedestrians, frame) 
                    bboxesInteresting = filter(lambda c: int(c[5])==0 or int(c[5])==1 , bboxes)
                    bboxesInteresting2 = copy.deepcopy(bboxesInteresting)
                    bboxesInterestingList = list(bboxesInteresting2)
                    image = utils.draw_bbox(frame, bboxesInteresting, classes=classesPedestrians)           
                    #and continue with the tracking   
                    detections = [] # llista d'objectes Detection per a cada frame

                    #print(bboxesInterestingList)                
                    for t in bboxesInterestingList:
                        startX=int(t[0])
                        startY=int(t[1])
                        endX=int(t[2])
                        endY=int(t[3])
                        classDetected = int(t[5])               
                        #print("hola")
                        #print (startX,startY,endX,endY)
                        centroid = ft.compute_centroid_at_floor(startX, startY, endX, endY)                  
                        detections.append(Detection(centroid))
                        detectionsWithClassAndwidth.append([frameIndex, centroid, classDetected, abs(startX-endX)])
                        
                    tracked_objects = tracker.update(detections=detections)
                    #print(tracked_objects)
                    #print(detections)
                    #draw_points(image, detections, radius=5, color=[0,255,255])
                    ft.draw_tracked_objects(image, tracked_objects, color=[0,255,255])
                    pointsWithId = ft.getTrackedObjectsCoordinates(tracked_objects)
                    #print (pointsWithId)
                    if len(pointsWithId):
                        trajectoryPoints.append([frameIndex,pointsWithId])

                    curr_time = time.time()
                    exec_time = curr_time - prev_time

                    #Let's draw the zebra passes.
                    for i in range(len(zebraPassCoordinates)):
                        c1, c2 = (int(zebraPassCoordinates[i][0]), int(zebraPassCoordinates[i][1])), (int(zebraPassCoordinates[i][2]), int(zebraPassCoordinates[i][3]))
                        cv2.rectangle(image, c1, c2, [255,0,0],2)
                    
                    #result = np.asarray(image)
                    info = "time: %.2f ms" % (1000 * exec_time)
                    #//cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)

                    #result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  #if we modify the channels we do not see it properly
                    #//imageResized = cv2.resize(image, dsize=(800,600), interpolation = cv2.INTER_CUBIC)
                    #//cv2.imshow("result", imageResized)
                    frameIndex = frameIndex + 1
                    print("Frame index is", frameIndex)
                    #//if cv2.waitKey(1) & 0xFF == ord('q'):
                    #//    break            
                else:
                    print("Finish processing!")
                    #raise ValueError("No image!")
                    break

            print("STEP 2: Done!")

        
            # STEP 3 -----------------------------------------------------

            #Now let's select the interesting detections from all the detections(we only keep the ones into the zebra pass and near it)
            interestingDetectionsForEveryZebraCrossArea, marginForEveryZebraCrossArea = ft.getInterestingDetecionsForEveryZebraCrossArea(zebraPassCoordinates, trajectoryPoints, detectionsWithClassAndwidth)

            #Let's calculate the centers of the waiting areas.
            waitingToCrossAreasCoordinates = ft.locateWaitingToCrossAreas(zebraPassCoordinates, interestingDetectionsForEveryZebraCrossArea)

            #Let's create a summary Image that shows the trajectories,the zebra cross areas and the waiting areas for each zebra pass
            resultTrajectories = np.asarray(imageToShowTrajectories)
            #//cv2.namedWindow("resultTrajectories", cv2.WINDOW_AUTOSIZE)

            #First we draw all the detections.
            for interestingDetectionsForEachZebraCrossArea in interestingDetectionsForEveryZebraCrossArea:
                colorAux = list(np.random.randint(255, size=3))
                color = [int(colorAux[0]), int(colorAux[1]), int(colorAux[2])]

            for eachFrame in interestingDetectionsForEachZebraCrossArea:
                for eachDetection in eachFrame[1]:
                    center = [int(eachDetection[1][0]), int(eachDetection[1][1])]
                    cv2.circle(resultTrajectories, center,5,color,3)

            #Second we draw the zebra passes.
            for i in range(len(zebraPassCoordinates)):
                c1, c2 = (int(zebraPassCoordinates[i][0]), int(zebraPassCoordinates[i][1])), (int(zebraPassCoordinates[i][2]), int(zebraPassCoordinates[i][3]))
                cv2.rectangle(resultTrajectories, c1, c2, [255,0,0],2)

            #Third we draw the waiting Areas
            cv2.circle(resultTrajectories, (waitingToCrossAreasCoordinates[i][0][0], waitingToCrossAreasCoordinates[i][0][1]),marginForEveryZebraCrossArea[i],[0,128,255],3)
            cv2.circle(resultTrajectories, (waitingToCrossAreasCoordinates[i][1][0], waitingToCrossAreasCoordinates[i][1][1]),marginForEveryZebraCrossArea[i],[0,128,255],3)

            #Finally we show the image          
            #//imageResizedTrajectories= cv2.resize(resultTrajectories, dsize=(800,600), interpolation = cv2.INTER_CUBIC)
            #//cv2.imshow("resultTrajectories", imageResizedTrajectories)

            #//if cv2.waitKey(100000) & 0xFF == ord('q'):
            #//  print ("Image has already been closed")

            statisticsToReturnInJson = ft.getStatisticsInSeconds(interestingDetectionsForEveryZebraCrossArea, detectionsWithClassAndwidth, zebraPassCoordinates, waitingToCrossAreasCoordinates, marginForEveryZebraCrossArea, fps)
            print (statisticsToReturnInJson)

            #//for n in range(len(statisticsToReturnInJson)):
            #//    #[NPersonsCrossingDir1, NByciclesCrossingDir1, avgTimeCrossingDir1, maxTimeCrossingDir1, minTimeCrossingDir1, #NPersonsCrossingDir2, NByciclesCrossingDir2, avgTimeCrossingDir2, maxTimeCrossingDir2, minTimeCrossingDir2, avgTimeWaitingWA1, #maxTimeWaitingWA1, minTimeWaitingWA1, avgTimeWaitingWA2, maxTimeWaitingWA2, minTimeWaitingWA2]
            #//    print ("STATISTICS FOR ZEBRA PASS", n)
            #//    print (" NPersons Crossing in Direction 1", statisticsToReturnInJson[n][0])
            #//    print (" NBycicles Crossing in Direction 1", statisticsToReturnInJson[n][1])
            #//    print (" avgTime Crossing in Direction 1 (seconds)", statisticsToReturnInJson[n][2])
            #//    print (" maxTime Crossing in Direction 1 (seconds)", statisticsToReturnInJson[n][3])
            #//    print (" minTime Crossing in Direction 1 (seconds)", statisticsToReturnInJson[n][4])
            #//    print (" NPersons Crossing in Direction 2", statisticsToReturnInJson[n][5])
            #//    print (" NBycicles Crossing in Direction 2", statisticsToReturnInJson[n][6])
            #//    print (" avgTime Crossing in Direction 2 (seconds)", statisticsToReturnInJson[n][7])
            #//    print (" maxTime Crossing in Direction 2 (seconds)", statisticsToReturnInJson[n][8])
            #//    print (" minTime Crossing in Direction 2 (seconds)", statisticsToReturnInJson[n][9])
            #//    print (" AvgTime Waiting in Waiting Area 1 (seconds)", statisticsToReturnInJson[n][10])
            #//    print (" maxTime Waiting in Waiting Area 1 (seconds)", statisticsToReturnInJson[n][11])
            #//    print (" minTime Waiting in Waiting Area 1 (seconds)", statisticsToReturnInJson[n][12])
            #//    print (" AvgTime Waiting in Waiting Area 2 (seconds)", statisticsToReturnInJson[n][13])
            #//    print (" maxTime Waiting in Waiting Area 2 (seconds)", statisticsToReturnInJson[n][14])
            #//    print (" minTime Waiting in Waiting Area 2 (seconds)", statisticsToReturnInJson[n][15])
            #//    print (" ")

            #//    pathAndNameForJsonFile = "./Statistics.json"
            #//    ft.buildAndSaveJsonFile(videoInformation, statisticsToReturnInJson, interestingDetectionsForEveryZebraCrossArea, pathAndNameForJsonFile)

            print("STEP 3: Done!")

            
            # RESULTS ----------------------------------------------------
            output = ft.buildJson(videoInformation, statisticsToReturnInJson, interestingDetectionsForEveryZebraCrossArea)
            results = json.dumps(output, indent=2, sort_keys=False)
            print("result = {}".format(results), file=sys.stderr)

            # return results in file
            pathname = os.path.dirname(message_data[QUEUE_TAG_VIDEO])
            filename = os.path.join(pathname, message_data[QUEUE_TAG_ID]+".json")
            with open(filename, "w") as filejson:
                json.dump(output, filejson, indent=2)    

        except:
            db_queue.set(message_data[QUEUE_TAG_ID], '""')

        # sleep for a small amount
        time.sleep(SERVER_SLEEP)
