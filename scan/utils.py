'''Richard HD'''

from imutils.perspective import four_point_transform, order_points
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils.video import FPS
import time
from sklearn.cluster import DBSCAN
import os

### SHAPE ANALYSIS ###

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def filter_approx(pts):
    """
    return the poins if the rectangle is valid and if its faced to the camerta
    """
    
    '''verify if the rectangle is not too long'''
    (x, y, w, h) = cv2.boundingRect(pts)
    ar = w / float(h) if w>h else h / float(w)
    if ar < 4:
        
        '''faced to the camera'''
        rect = order_points(pts.reshape(4, 2))
        (tl, tr, br, bl) = rect
        intercection = line_intersection((tl, br), (tr, bl))
        if intercection != None: # verify the intercection

            intercection = np.array(intercection)
            dist = [np.linalg.norm(intercection-tl), np.linalg.norm(intercection-br)]
            distA, distB = max(dist), min(dist)

            dist = [np.linalg.norm(intercection-tr), np.linalg.norm(intercection-bl)]
            distC, distD = max(dist), min(dist)
            
            if distA/distB < 2. and distC/distD < 2.:   

                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                minWidth = min(int(widthA), int(widthB)) + 0.00001

                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                minHeight = max(int(heightA), int(heightB)) + 0.00001

                if(maxWidth/minWidth<2.5 and maxHeight/minHeight<2.5): return pts
        
def detect_document(image):
    """
    Extracts tries to extrat the document location
    """
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 55, 200)
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
        for c in cnts:
            '''approximate the contour'''
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            '''if our approximated contour has four points, then we can assume that we have found our screen'''
            if len(approx) == 4:
                return filter_approx(approx)
            

### FRAME ANALYSIS ###

def not_valid_shape(shape):
    return max(shape)/min(shape) > 4

def scan_from_frame(frame):
    '''
    crops the document from each frame
    '''
    screenCnt = detect_document(frame)
    if not(np.any(screenCnt)):
        return None
    
    warped = four_point_transform(frame, screenCnt.reshape(4, 2))
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    '''filter by shape'''
    if not_valid_shape(warped.shape):
        cv2.drawContours(frame, [screenCnt], -1, (0, 0, 255), 2)
    else:     
        cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)
        return warped


### FILTER IMAGES ###

def dbscan_filter(scans,  min_samples=50):
    '''filter the noise on scaned images'''
    shapes = [scan.shape for scan in scans]
    db = DBSCAN(eps=20, min_samples=min_samples, metric='euclidean').fit(shapes)
    labels = db.labels_
    def most_common(lst):
        return max(set(lst), key=list(lst).count)
    most_common_label = most_common(labels)
    return [scan for label,scan in zip(labels,scans) if label==most_common_label and label>-1]