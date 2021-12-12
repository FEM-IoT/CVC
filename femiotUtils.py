import os
import sys
import cv2
import time
import copy
import math
import shapely
import json
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
from collections import Counter
from statistics import mean
from shapely.geometry import LineString, Point
from norfair import Detection, Tracker, Video, draw_tracked_objects, draw_points	

#We define the distance function to calculate the correspoding points between frames.
def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)
 
#To calculate the centroid of the bounding box.
def compute_centroid_at_floor(startX, startY, endX, endY):   
    cx = int(startX + (endX-startX)/2)
    cy = int(endY)
    return np.array([cx,cy])

def getTrackedObjectsCoordinates(objects):
    pointsToReturn=[]
    for obj in objects:
      if not obj.live_points.any():
        continue
      for point, live in zip(obj.estimate, obj.live_points):
        if live:
          #print (point.astype(int))
          pointsToReturn.append([obj.id,point.astype(float)])
          #print(tuple(point.astype(int)))
          #print (obj.id)
    return pointsToReturn

#To initialize the Yolo
def initTensors(pb_file, graph):
  return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
  return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
  return return_tensors

def inferenceTensors(sess, return_tensors, num_classes, score_thresh, frame):
  input_size = 416
  #iou_type = 'diou' if 'yolov4' in pb_file else 'giou' #yolov4:diou, else giou
  iou_type = 'giou' #yolov4:diou, else giou
  iou_thresh = 0.35

  image = Image.fromarray(frame)
  frame_size = frame.shape[:2]
  image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
  image_data = image_data[np.newaxis, ...]

  pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]], 
              feed_dict={return_tensors[0]: image_data})

  pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)), np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)                

  bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, score_thresh)
  bboxes = utils.nms(bboxes, iou_type, iou_thresh, method='nms')
  return bboxes

def getZebraPassAvgCoordinates(zebraPassesDetected):
  #Let's determine the Zebra Passes detected based on the detections:
  #print (type(zebraPassesDetected))
  #print(zebraPassesDetected)
  #Looks in the majority of frames how many zebra passes where detected
  mostVotedNumberOfZebraPasses=Counter(list(map(len,zebraPassesDetected))).most_common(1) 
  nSupposedZebraPasses = mostVotedNumberOfZebraPasses[0][0]
  print("nSupposedZebraPasses",nSupposedZebraPasses)
  
  zebraDetectionsFiltered = []
  zebraDetectionsFilteredMasCenters = []

  for i in zebraPassesDetected: #we loop for each frame
    for j in i:  #we loop for each detecion
      coor = np.array(j[:4], dtype=np.int32) #x_min, y_min, x_max, y_max
      massCenterForCurrentDetection = [(coor[0] + coor[2])/2.0, (coor[1] + coor[3])/2.0]
      addAsNewZebraPassZone = True
      for n in range(len(zebraDetectionsFilteredMasCenters)):
        nItem = zebraDetectionsFilteredMasCenters[n]
        if abs(massCenterForCurrentDetection[0]-nItem[0])<50 and abs(massCenterForCurrentDetection[1]-nItem[1])<50:  
	#if the center of masses are neararer than a certain distance, we consider them the same zebra pass and join them.
          addAsNewZebraPassZone = False
          zebraDetectionsFiltered[n].append(coor)
      if addAsNewZebraPassZone:
        zebraDetectionsFiltered.append([coor])
        zebraDetectionsFilteredMasCenters.append(massCenterForCurrentDetection)
            
  #print(zebraDetectionsFilteredMasCenters)
  #print(zebraDetectionsFiltered)

  #print(len(zebraDetectionsFilteredMasCenters))
  #print(len(zebraDetectionsFiltered))
  #print(list(map(len,zebraDetectionsFiltered)))
  arr = np.array(list(map(len,zebraDetectionsFiltered)))
  indexOfMaximumZebraPasses = (-arr).argsort()[:nSupposedZebraPasses]
  #print(indexOfMaximumZebraPasses)

  zebraPassCoordinates=[]
  for i in indexOfMaximumZebraPasses:
    zebraPassCoordinatesIter = np.mean(zebraDetectionsFiltered[i],axis=0)
    zebraPassCoordinates.append(zebraPassCoordinatesIter)
  return zebraPassCoordinates

def getInterestingDetecionsForEveryZebraCrossArea(zebraPassCoordinates, trajectoryPoints, detectionsWithClassAndwidth):
#Let's determine wich are the interesting detections for every zebra Passs
  interestingDetectionsForEveryZebraCrossArea=[]
  marginForEveryZebraCrossArea = []
  for everyZebraCrossArea in zebraPassCoordinates:
    interestingDetectionsForCurrrentZebraCrossArea,margin = filterTrajectoriesPassingThroughAZebraPass(everyZebraCrossArea, trajectoryPoints,detectionsWithClassAndwidth)
    #if (len(interestingDetectionsForCurrrentZebraCrossArea) != 0):
    interestingDetectionsForEveryZebraCrossArea.append(interestingDetectionsForCurrrentZebraCrossArea)
    marginForEveryZebraCrossArea.append(margin)
  return interestingDetectionsForEveryZebraCrossArea,marginForEveryZebraCrossArea

def filterTrajectoriesPassingThroughAZebraPass(zebraCrossArea, trajectoryPoints, detectionsWithClassAndwidth):
   #ZebraCrossArea [x_min, y_min, x_max, y_max]
   #trajectoryPoints [frame,[objectId, (x,y)]]mean
   #detectionsWithClassAndwidth[frameIndex, centroid, classDetected, width]
   #print ("DETECTIONS WITH CLASS AND WIDTH", detectionsWithClassAndwidth)
    
   #First we have to decide which is the margin, which is specific for every zebra pass. It will be the average
   #of all the detections into a zebra pass.
   detectionsWidthIntoTheZebraPass=[]
   for detection in detectionsWithClassAndwidth:
      #detecion = [classId, centroid, classDetected, width]
      #print ("Detection is", detection)
      #print ("detection 2 is:",detection[2]) 
      if (detection[2]==0): #we only consider the detections of pedestrians for the margin
        xDetection = detection[1][0]
        yDetection = detection[1][1]
      if (xDetection > zebraCrossArea[0]) and (xDetection < zebraCrossArea[2]):
        if (yDetection > zebraCrossArea[1]) and (yDetection < zebraCrossArea[3]):
          detectionsWidthIntoTheZebraPass.append(detection[3])
   if (len(detectionsWidthIntoTheZebraPass) > 0):  #margin is 2.5 times the average detection of a person
     margin = mean(detectionsWidthIntoTheZebraPass) * 2.5
   else: #if there are no detections the margin will be the minimum size (width or height) of the zebra pass
     margin = 50 #initialization
     width = abs(zebraCrossArea[2]-zebraCrossArea[0])
     height = abs(zebraCrossArea[3]-zebraCrossArea[1])
     if (width<height):
       margin = width
     else:
       margin = height
   print ("margin is",margin)

   #If the trajectory point is passing through the zebra pass, we select it, with a certain margin.
   interestingDetectionsForCurrrentZebraCrossArea=[]
   for detectionsInAFrame in trajectoryPoints:
      interstingDetectionForCurrentFrame=[]
      frameId = detectionsInAFrame[0]
      for detection in detectionsInAFrame[1]:
        #detecion = [objectId, [x,y]]
        ojectId= detection[0]
        xDetection = detection[1][0]
        yDetection = detection[1][1]
        if (xDetection > zebraCrossArea[0] - margin) and (xDetection < zebraCrossArea[2] + margin):
          if (yDetection > zebraCrossArea[1] - margin) and (yDetection < zebraCrossArea[3] + margin):
            interstingDetectionForCurrentFrame.append(detection)
      if (len(interstingDetectionForCurrentFrame) != 0):
        interestingDetectionsForCurrrentZebraCrossArea.append([frameId, interstingDetectionForCurrentFrame])
      
   return interestingDetectionsForCurrrentZebraCrossArea,int(margin)

def locateWaitingToCrossAreas(zebraPassCoordinates, interestingDetectionsForEveryZebraCrossArea):
  waitingAreaCenterForAllZebraPasses=[]
  #print("ID",interestingDetectionsForEveryZebraCrossArea)

  for n in range(len(zebraPassCoordinates)):
    everyZebraCrossArea = zebraPassCoordinates[n]
    #ZebraCrossArea [x_min, y_min, x_max, y_max]
    [x_min, y_min, x_max, y_max] = everyZebraCrossArea
    if (abs(x_max - x_min) >   abs(y_max - y_min)):  #this is the normal case where the zebra pass is horitzontal
      candidatesForWaitingArea = [[int(x_min), int(y_min) + int(abs(y_max - y_min)/2)],[int(x_max), int(y_min) + int(abs(y_max - y_min)/2)]]
    else:
      candidatesForWaitingArea = [[int(x_min) + int(abs(x_max - x_min) / 2), int(y_min)],[int(x_min) + int(abs(x_max - x_min) / 2), int(y_max)]]

    #Now let's take all the interesting points of the trajectories OUTSIDE the zebra pass area, and assign them to one of initial points. After that we will calculate the center of masses of the clusters and this will be the center of the waiting area.
    interestingDetectionsForWaitingAreas = [[],[]]
    #print ("Interestng detection sfor waiting Areas",  interestingDetectionsForEveryZebraCrossArea[n])
    for eachFrame in interestingDetectionsForEveryZebraCrossArea[n]:
      for eachDetection in eachFrame[1]:
        #detecion = [objectId, (x,y)]
        #print ("detection",eachDetection)
        xPosition = eachDetection[1][0]
        yPosition = eachDetection[1][1]
        #print("Xy positions", xPosition, yPosition)
        #print("zebra area",x_min, y_min, x_max, y_max)
        #if the point is into the Zebra pass, we discard it
        if not (xPosition > x_min and xPosition < x_max) and (yPosition > y_min and yPosition < y_max):          
          #print (xPosition,yPosition)
          #now we have to determine to which of the candidate points is more near.
          dist1 = math.sqrt(pow((candidatesForWaitingArea[0][0] - xPosition),2) + pow((candidatesForWaitingArea[0][1] - yPosition),2))
          dist2 = math.sqrt(pow((candidatesForWaitingArea[1][0] - xPosition),2) + pow((candidatesForWaitingArea[1][1] - yPosition),2))
          #print("dists",dist1,dist2)
          if (dist1 < dist2):
            interestingDetectionsForWaitingAreas[0].append([xPosition, yPosition])
          else:
            interestingDetectionsForWaitingAreas[1].append([xPosition, yPosition])

      #now that every detecion is assigned to one candidate, let's move the candidates finding the center of masses (with respect to the y coordinate).
    if(len(interestingDetectionsForWaitingAreas[0]) > 0):
      xValues0 = [item[0] for item in interestingDetectionsForWaitingAreas[0]]
      yValues0 = [item[1] for item in interestingDetectionsForWaitingAreas[0]]   
      xValuesAverage0 = mean(xValues0)
      yValuesAverage0 = mean(yValues0)
    else:  #if we haven't found any trajectories, we use the central point of the zebra pass
      xValuesAverage0 = candidatesForWaitingArea[0][0]
      yValuesAverage0 = candidatesForWaitingArea[0][1]

    if(len(interestingDetectionsForWaitingAreas[1]) > 0):
      xValues1 = [item[0] for item in interestingDetectionsForWaitingAreas[1]]
      yValues1 = [item[1] for item in interestingDetectionsForWaitingAreas[1]]   
      xValuesAverage1 = mean(xValues1)
      yValuesAverage1 = mean(yValues1)
      candidatesForWaitingArea[1][0]=int(xValuesAverage1)
      candidatesForWaitingArea[1][1]=int(yValuesAverage1)
    else:  #if we haven't found any trajectories, we use the central point of the zebra pass
      xValuesAverage1 = candidatesForWaitingArea[1][0]
      yValuesAverage1 = candidatesForWaitingArea[1][1]

    #now we will draw a line from one center of mass to the other, and find the intersection point with the zebra area pass.
    #The intersection can be with any of the 4 sides. This will be the point where the waiting area starts.
    lineTrajectories = LineString([(xValuesAverage0, yValuesAverage0), (xValuesAverage1, yValuesAverage1)])
    lineZebraPass1 =  LineString([(x_min, y_min), (x_max, y_min)])
    lineZebraPass2 =  LineString([(x_max, y_min), (x_max, y_max)])
    lineZebraPass3 =  LineString([(x_max, y_max), (x_min, y_max)])
    lineZebraPass4 =  LineString([(x_min, y_max), (x_min, y_min)])

    waitingAreaPoints=[]
    intersectionPoint1 = lineTrajectories.intersection(lineZebraPass1)
    #print(intersectionPoint1)
    if (not intersectionPoint1.is_empty):
      waitingAreaPoints.append([int(intersectionPoint1.x), int(intersectionPoint1.y)]) 

    intersectionPoint2 = lineTrajectories.intersection(lineZebraPass2)
    #print(intersectionPoint2)
    if (not intersectionPoint2.is_empty):
      waitingAreaPoints.append([int(intersectionPoint2.x), int(intersectionPoint2.y)]) 

    intersectionPoint3 = lineTrajectories.intersection(lineZebraPass3)
    #print(intersectionPoint3)
    if (not intersectionPoint3.is_empty):
      waitingAreaPoints.append([int(intersectionPoint3.x), int(intersectionPoint3.y)]) 

    intersectionPoint4 = lineTrajectories.intersection(lineZebraPass4)
    #print(intersectionPoint4)
    if (not intersectionPoint4.is_empty):
      waitingAreaPoints.append([int(intersectionPoint4.x), int(intersectionPoint4.y)]) 

    if(len(waitingAreaPoints) != 2):
      print("Not enough points to determine waiting areas. Leaving the default ones!!!")
    else:
      candidatesForWaitingArea[0]=waitingAreaPoints[0]
      candidatesForWaitingArea[1]=waitingAreaPoints[1]
    waitingAreaCenterForAllZebraPasses.append(candidatesForWaitingArea)

  return waitingAreaCenterForAllZebraPasses

def getStatisticsInFramesForZebraPass(summarizedTrajectoriesZebraPassDirection):
  #Let's calculate the statistics: 
  NPersonsCrossing = 0
  NBiciclesCrossing = 0
  nFramesForCurrentDetection = []
  LengthsForCurrentDetection = []
  maxLenghth = 0
  for eachDetecion in summarizedTrajectoriesZebraPassDirection:
    if (eachDetecion[1] == 0): #Considering the persons' detections
      if (eachDetecion[7] > maxLenghth):
        maxLenghth = eachDetecion[7]
      nFramesForCurrentDetection.append(eachDetecion[3] - eachDetecion[2])
      LengthsForCurrentDetection.append(eachDetecion[7])
      NPersonsCrossing = NPersonsCrossing + 1
    elif (eachDetecion[1] == 1): #when detecting a bicicle is also detecting a person that we do not want to contabilize.
      NBiciclesCrossing = NBiciclesCrossing + 1
      NPersonsCrossing = NPersonsCrossing -1
         
  #in order to avoid problems with problems with the tracking, to calculate the times we will only consider the 
  #detections that have a lenght of 75% of the maximum
  nFramesForCurrentDetectionFiltered = []
  lengthsFiltered = []
  #print ("Lengths for current Detection",LengthsForCurrentDetection)
  for n in range(len(LengthsForCurrentDetection)):
    if (LengthsForCurrentDetection[n] > maxLenghth * 0.75):
      lengthsFiltered.append(LengthsForCurrentDetection[n])
      nFramesForCurrentDetectionFiltered.append(nFramesForCurrentDetection[n])

  #print ("Lengths for current Detection Filtered",lengthsFiltered)
  #print ("Frames filtered", nFramesForCurrentDetectionFiltered)
  if (len(nFramesForCurrentDetectionFiltered) >0):
    avgTimeCrossing = mean(nFramesForCurrentDetectionFiltered)
    maxTimeCrossing = max(nFramesForCurrentDetectionFiltered)
    minTimeCrossing = min (nFramesForCurrentDetectionFiltered)
  else:
   avgTimeCrossing = 0
   maxTimeCrossing = 0
   minTimeCrossing = 0

  return [NPersonsCrossing, NBiciclesCrossing, avgTimeCrossing, maxTimeCrossing, minTimeCrossing]

def getStatisticsInFramesForWaitingArea(summarizedTrajectoriesWaitingArea):
  nFramesForCurrentDetection = []
  for eachDetecion in summarizedTrajectoriesWaitingArea:
      nFramesForCurrentDetection.append(eachDetecion[3] - eachDetecion[2])
  
  #we do not need to filter these detections because the areas are very little.
  if (len (nFramesForCurrentDetection) >0):  
    avgTimeWaiting = mean(nFramesForCurrentDetection)
    maxTimeWaiting = max(nFramesForCurrentDetection)
    minTimeWaiting = min (nFramesForCurrentDetection)
  else:
    avgTimeWaiting =0
    maxTimeWaiting =0
    minTimeWaiting =0

  return [avgTimeWaiting, maxTimeWaiting, minTimeWaiting]

def getStatisticsInSeconds(interestingDetectionsForEveryZebraCrossArea, detectionsWithClassAndwidth, zebraPassCoordinates, waitingToCrossAreasCoordinates, marginForEveryZebraCrossArea,fps):
#This function is getting all the interesting (near) detections, their class detected, all the ZebraPassAreas, the coordinates #where every waiting area starts, and the size of the considered waiting area.
#It has to determine each object how many frames is present into each zebra pass and waiting area, and count those objects.
#Values that should return for every Zebra Pass:
#NPersons Crossing in direction 1
#NBycicles Crossing in direction 1
#NPersons Crossing in direction 2
#NBycicles Crossing in direction 2
#Avg Time Crossing in direction 1 in Frames
#Max Time Crossing in direction 1 in Frames
#Min Time Crossing in direction 1 in Frames
#Avg Time Crossing in direction 2 in Frames
#Max Time Crossing in direction 2 in Frames
#Min Time Crossing in direction 2 in Frames
#Avg Time Waiting in direction 1 in Frames
#Max Time Waiting in direction 1 in Frames
#Min Time Waiting in direction 1 in Frames
#Avg Time Waiting in direction 2 in Frames
#Max Time Waiting in direction 2 in Frames
#Min Time Waiting in direction 2 in Frames

  detectionsWithClassAndwidth = detectionsWithClassAndwidth
  #print("detectionsWithClassAndwidth",currentDetectionsWithClassAndwidth)
  #detectionsWithClassAndwidth = [[nFrame, array(x,y), class, width],[nFrame, array(x,y), class, width],[nFrame, array(x,y), class, width]] 

  statisticsToReturn = []
  for n in range(len(zebraPassCoordinates)):
    currentZebraCrossArea = zebraPassCoordinates[n]
    [x_min, y_min, x_max, y_max] = currentZebraCrossArea
    marginForCurrentZebraCrossArea = marginForEveryZebraCrossArea[n]
    #print("currentMargin",marginForCurrentZebraCrossArea)
    currentCoordinatesForWaitingCrossAreas = waitingToCrossAreasCoordinates[n]
    # coordinatesForWaitingCrossAreas = [[x1,y1],[x2,y2]]
    #print("coordinatesForWaitingCrossAreas",coordinatesForWaitingCrossAreas)
    currentInterestingDetections=interestingDetectionsForEveryZebraCrossArea[n]
    #currentInterestingDetections = [[nFrame,[[objectId, array(x,y)],[objectId, array(x,y)]]],[nFrame,[[objectId, array(x,y)],[objectId, array(x,y)]]]]
    #print("currentInterestingDetections",currentInterestingDetections)

    [summarizedTrajectoriesZebraPassDirection1, summarizedTrajectoriesZebraPassDirection2, summarizedTrajectoriesWaitingArea1, summarizedTrajectoriesWaitingArea2] = getSummarizedTrajectoriesForZebraAndWaitingAreas(currentInterestingDetections, detectionsWithClassAndwidth, currentZebraCrossArea, currentCoordinatesForWaitingCrossAreas, marginForCurrentZebraCrossArea)
    print ("Trajectories summarized for Zebra pass Dir 1", n, summarizedTrajectoriesZebraPassDirection1)
    print ("Trajectories summarized for Zebra pass Dir 2", n, summarizedTrajectoriesZebraPassDirection2)
    print ("Trajectories summarized for waiting Area 1", n, summarizedTrajectoriesWaitingArea1)
    print ("Trajectories summarized for waiting Area 2", n, summarizedTrajectoriesWaitingArea2)

    [NPersonsCrossingDir1, NByciclesCrossingDir1, avgTimeCrossingDir1, maxTimeCrossingDir1, minTimeCrossingDir1] = getStatisticsInFramesForZebraPass(summarizedTrajectoriesZebraPassDirection1)
    #print ("Statistics Dir 1 are:",NPersonsCrossingDir1, NByciclesCrossingDir1, avgTimeCrossingDir1, maxTimeCrossingDir1, minTimeCrossingDir1)

    [NPersonsCrossingDir2, NByciclesCrossingDir2, avgTimeCrossingDir2, maxTimeCrossingDir2, minTimeCrossingDir2] = getStatisticsInFramesForZebraPass(summarizedTrajectoriesZebraPassDirection2)
    #print ("Statistics Dir 2 are:",NPersonsCrossingDir2, NByciclesCrossingDir2, avgTimeCrossingDir2, maxTimeCrossingDir2, minTimeCrossingDir2)

    [avgTimeWaitingWA1, maxTimeWaitingWA1, minTimeWaitingWA1] = getStatisticsInFramesForWaitingArea(summarizedTrajectoriesWaitingArea1)
    #print ("Statistics WA 1 are:",avgTimeWaitingWA1, maxTimeWaitingWA1, minTimeWaitingWA1)

    [avgTimeWaitingWA2, maxTimeWaitingWA2, minTimeWaitingWA2] = getStatisticsInFramesForWaitingArea(summarizedTrajectoriesWaitingArea2)
    #print ("Statistics WA 2 are:",avgTimeWaitingWA2, maxTimeWaitingWA2, minTimeWaitingWA2)

    statisticsToReturn.append([NPersonsCrossingDir1, NByciclesCrossingDir1, avgTimeCrossingDir1/fps, maxTimeCrossingDir1/fps, minTimeCrossingDir1/fps, NPersonsCrossingDir2, NByciclesCrossingDir2, avgTimeCrossingDir2/fps, maxTimeCrossingDir2/fps, minTimeCrossingDir2/fps, avgTimeWaitingWA1/fps, maxTimeWaitingWA1/fps, minTimeWaitingWA1/fps, avgTimeWaitingWA2/fps, maxTimeWaitingWA2/fps, minTimeWaitingWA2/fps])

  return statisticsToReturn

def buildJson(videoInformation, statisticsToReturnInJson, interestingDetectionsForEveryZebraCrossArea):
  # VideoInformation = [fps, width, height, frameCount]
  duration = float(videoInformation[3]) / float(videoInformation[0])
  width = videoInformation[1]
  height = videoInformation[2]
  fps = float(videoInformation[0])

  outputjson = { }
  outputjson["VideoInformation"] = { "Duration":duration, "Width":width, "height":height, "Fps":fps, "NZebraCrossingAreas":len(statisticsToReturnInJson) }
          
  #for each zebra pass we write the statistics and detections
  outputjson["ZebraCrossingAreasResults"] = []
  for n in range(len(statisticsToReturnInJson)): 
    serializableDetections=[]
    for everyFrame in interestingDetectionsForEveryZebraCrossArea[n]:
      detectionsForThisFrame=[]
      frameId = everyFrame[0]
      for everyDetection in everyFrame[1]:
        objectId = everyDetection[0]
        x = everyDetection[1][0]
        y = everyDetection[1][1]
        detection = [objectId, x, y]
        detectionWithFrame = np.append(detection,frameId)
        serializableDetections.append(detectionWithFrame.tolist())
          
    #zebra crossing statistics and detections
    outputjson_data = {
      "ZebraCrossingAreaId":n,
      "NPersonsCrossingDirA":statisticsToReturnInJson[n][0], "NByciclesCrossingDirA":statisticsToReturnInJson[n][1], "AvgTimeCrossingDirA": statisticsToReturnInJson[n][2], "MaxTimeCrossingDirA":statisticsToReturnInJson[n][3], "MinTimeCrossingDirA":statisticsToReturnInJson[n][4],
      "NPersonsCrossingDirB":statisticsToReturnInJson[n][5], "NByciclesCrossingDirB":statisticsToReturnInJson[n][6], "AvgTimeCrossingDirB": statisticsToReturnInJson[n][7], "MaxTimeCrossingDirB":statisticsToReturnInJson[n][8], "MinTimeCrossingDirB":statisticsToReturnInJson[n][9], "AvgTimeWaitingAreaA": statisticsToReturnInJson[n][10],
      "MaxTimeWaitingAreaA":statisticsToReturnInJson[n][11], "MinTimeWaitingAreaA":statisticsToReturnInJson[n][12],
      "AvgTimeWaitingAreaB":statisticsToReturnInJson[n][13], "MaxTimeWaitingAreaB":statisticsToReturnInJson[n][14],
      "MinTimeWaitingAreaB":statisticsToReturnInJson[n][13],
      "Detections":serializableDetections
    }

    outputjson["ZebraCrossingAreasResults"].append(outputjson_data)
  
  JSON_TAG_RESULTS = "results"
  output = { JSON_TAG_RESULTS : [] }
  output[JSON_TAG_RESULTS].append(outputjson)

  results = json.dumps(output, indent=2, sort_keys=False)
  print("result = {}".format(results), file=sys.stderr)

  #return results
  return output

def buildAndSaveJsonFile(videoInformation, statisticsToReturnInJson, interestingDetectionsForEveryZebraCrossArea, pathAndNameForJsonFile):
    #build json result
    output = buildJson(videoInformation, statisticsToReturnInJson, interestingDetectionsForEveryZebraCrossArea)

    #return results in file
    filename = pathAndNameForJsonFile
    with open(filename, "w") as filejson:
      json.dump(output, filejson, indent=2)
 
def getClassFromPosition(position, detectionsWithClassAndwidth):
#position array(x,y)  
#detectionsWithClassAndwidth = [[nFrame, array(x,y), class, width],[nFrame, array(x,y), class, width],[nFrame, array(x,y), class, width]]
  classId = 0
  #print(detectionsWithClassAndwidth)
  #print(position)
  for eachDetecion in detectionsWithClassAndwidth:
    #print ("Detection", eachDetecion[1])
    #print ("Position", round(position[0]), round(position[1]))
    if (eachDetecion[1][0] == round(position[0])) and (eachDetecion[1][1] == round(position[1])):
      classId = eachDetecion[2]
      #print ("										Found", classId)
      break
  return classId

def addOrUpdateTrajectoryToAList(listToAddOrUpdate, objectId, classId, frameId, position):
  objectAlreadyPresent=0
  for summarizedTrajectory in listToAddOrUpdate:
    if (summarizedTrajectory[0] == objectId): #The object is already present (and so, detected in a previous frame)
       objectAlreadyPresent=1            
       if (frameId > summarizedTrajectory[3]):
       #if currentFrame is higher than last frame, we update it.
         summarizedTrajectory[3] = frameId 
       summarizedTrajectory[5] = position #we update the last position for the current object.

  if (not objectAlreadyPresent): #if the object was not present, we add it to the list of summarized trajectories.
    listToAddOrUpdate.append([objectId, classId, frameId, frameId, position, position])
  return listToAddOrUpdate

def getSummarizedTrajectoriesForZebraAndWaitingAreas(interestingDetections, detectionsWithClassAndwidth, zebraCrossArea, coordinatesForWaitingCrossAreas, marginForZebraCrossArea):
  #Given all the data it should return 3 lists (1 for the zebra crossing area, and 1 for each waiting area) with:
  #ObjectId, Class, FirstFrame, LastFrame, FistPostion, LastPosition, direction*, length*      (*) only for zebra pass detections
  
  #interestingDetections = [[nFrame,[[objectId, array(x,y)],[objectId, array(x,y)]]],[nFrame,[[objectId, array(x,y)],[objectId, array(x,y)]]]]
  #detectionsWithClassAndwidth = [[nFrame, array(x,y), class, width],[nFrame, array(x,y), class, width],[nFrame, array(x,y), class, width]]
  #zebraCrossArea = [x_min, y_min, x_max, y_max]
  #coordinatesForWaitingCrossAreas = [[x1,y1],[x2,y2]]
  
  summarizedTrajectoriesForZebraPassArea=[]
  summarizedTrajectoriesForWaitingArea1=[]
  summarizedTrajectoriesForWaitingArea2=[]
  
  [x_min, y_min, x_max, y_max] = zebraCrossArea
  for eachFrame in interestingDetections:
    frameId = eachFrame[0]
    for eachDetectionInFrame in eachFrame[1]:
      objectId = eachDetectionInFrame[0]
      position = eachDetectionInFrame[1]
      classId = getClassFromPosition(position, detectionsWithClassAndwidth)
      xPosition = position[0]
      yPosition = position[1]
      if ((xPosition >= x_min and xPosition <= x_max) and (yPosition >= y_min and yPosition <= y_max)):
        #the detection is in the Zebra pass
        #print ("Detection in Zebra Pass.")
        summarizedTrajectoriesForZebraPassArea = addOrUpdateTrajectoryToAList(summarizedTrajectoriesForZebraPassArea, objectId, classId, frameId, position)
      else:
        [x1,y1]=coordinatesForWaitingCrossAreas[0];
        [x2,y2]=coordinatesForWaitingCrossAreas[1];
        #calculathe the distance from the detection to the interesting areas centroids.
        dist1 = math.sqrt(pow((xPosition - x1),2) + pow((yPosition - y1),2))
        dist2 = math.sqrt(pow((xPosition - x2),2) + pow((yPosition - y2),2))
        if (dist1 < marginForZebraCrossArea):
          #print("Detection in Waiting Area 1.")
          summarizedTrajectoriesForWaitingArea1 = addOrUpdateTrajectoryToAList(summarizedTrajectoriesForWaitingArea1, objectId, classId, frameId, position)
        elif (dist2 < marginForZebraCrossArea):
          #print("Detection in Waiting Area 2.")
          summarizedTrajectoriesForWaitingArea2 = addOrUpdateTrajectoryToAList(summarizedTrajectoriesForWaitingArea2, objectId, classId, frameId, position)

  #Let's set the direction and length for the zebra pass detections
  summarizedTrajectoriesForZebraPassAreaWithDirectionAndLengthDirection1 = []
  summarizedTrajectoriesForZebraPassAreaWithDirectionAndLengthDirection2 = []
  for eachDetection in summarizedTrajectoriesForZebraPassArea:
    FirstPostion = eachDetection[4]
    LastPostion = eachDetection[5]
    #we calculate the distance of the first and last points of the detected object to the top left corner, to determine the direction
    dist1 = math.sqrt(pow((x_min - FirstPostion[0]),2) + pow((y_min - FirstPostion[1]),2))
    dist2 = math.sqrt(pow((x_min - LastPostion[0]),2) + pow((y_min - LastPostion[1]),2))
    if (dist1 < dist2):
      direction = 1
    else:
      direction = 2
    length = math.sqrt(pow((FirstPostion[0] - LastPostion[0]),2) + pow((FirstPostion[1] - LastPostion[1]),2))

    if (direction == 1):
      summarizedTrajectoriesForZebraPassAreaWithDirectionAndLengthDirection1.append([eachDetection[0], eachDetection[1], eachDetection[2], eachDetection[3], eachDetection[4], eachDetection[5], direction, length])
    elif (direction == 2):
      summarizedTrajectoriesForZebraPassAreaWithDirectionAndLengthDirection2.append([eachDetection[0], eachDetection[1], eachDetection[2], eachDetection[3], eachDetection[4], eachDetection[5], direction, length])

  return [summarizedTrajectoriesForZebraPassAreaWithDirectionAndLengthDirection1, summarizedTrajectoriesForZebraPassAreaWithDirectionAndLengthDirection2, summarizedTrajectoriesForWaitingArea1, summarizedTrajectoriesForWaitingArea2]

def getVideoInformation(vid):
  fps = vid.get(cv2.CAP_PROP_FPS)
  width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
  frameCount = vid.get(cv2.CAP_PROP_FRAME_COUNT)
  print ("The video has ", fps, "frames per second")
  print ("width of the video is ", width, "pixels")
  print ("height of the video is ", height, "pixels")
  print ("N Frames are: ",frameCount)
  videoInformation = [fps, width, height, frameCount]
  return videoInformation	
