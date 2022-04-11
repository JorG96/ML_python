# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:34:39 2022

@author: Jorge.Camacho1
"""

import numpy as np
import argparse
import time
import cv2
import os


ap=argparse.ArgumentParser()
ap.add_argument("-i", "--image",required=True,help="path to input image")
ap.add_argument("-y", "--yolo",required=True,help="base path to YOLO directory")
ap.add_argument("-c", "--confidence",type=float,default=0.5,help="minimum probability ti filter weak detections")
ap.add_argument("-t", "--threshold",type=float,default=0.3,help="threshold when apllying non-maxima suppression")
args= vars(ap.parse_args())
