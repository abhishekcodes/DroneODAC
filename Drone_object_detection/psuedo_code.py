# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:49:58 2019

@author: Durgesh
"""
#psuedo code for drone 

def compute_drone_action((x1,y1), (x2,y2)):
  #define the possible turning and moving action as strings
  turning = ""
  moving = ""
  raise = ""
  area, center = compute_area_and_center((x1,y1), (x2, y2))
  #obtain a x center between 0.0 and 1.0
  normalized_center[x] = center[x] / image.width
  #obtain a y center between 0.0 and 1.0  
  normalized_center[y] = center[y] / image.width
  if normalized_center[x] > 0.6 :
     turning = "turn_right"
  elif normalized_center[x] < 0.4 :
     turning = "turn_left"
  if normalized_center[y] > 0.6 :
    raise = "upwards"
  elif normalized_center[y] < 0.4 :
    raise = "downwards"
  #if the area is too big move backwards
  if area > 100 : 
    moving = "backwards" 
  elif area < 80 :
    moving = "ahead"
  return turning, moving, raise