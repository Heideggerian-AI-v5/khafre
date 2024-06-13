import ctypes 
import numpy
from numpy.ctypeslib import ndpointer 
import os
import platform
import cv2 as cv

from .contactcpp import _contact

dilationKernel = cv.getStructuringElement(cv.MORPH_CROSS,(3,3),(1,1))

def _contactSlow(searchWidth, threshold, imgHeight, imgWidth, dilImgS, dilImgO, maskImgS, maskImgO, depthImg, contPS, contPO):
    """
Should not call this. A naive implementation of searching for contact masks in python. Too slow for practical use,
present here to have a kind of easier to check reference and tester for the faster implementation in C++.
    """
    def _ci(searchWidth, threshold, imgHeight, imgWidth, dilImgD, maskImgD, maskImgN, contPN):
        #dKS = cv.getStructuringElement(cv.MORPH_RECT,(searchWidth*2+1,searchWidth*2+1),(searchWidth,searchWidth))
        #contPN = numpy.bitwise_and(dilImgD,maskImgN)
        contPN = numpy.array(maskImgN)
        return contPN
        dbg = set()
        for k in range(imgHeight):
            for j in range(imgWidth):
                if (0 < maskImgN[k][j]) and (0 < dilImgD[k][j]):
                    found = False
                    for l in range(-searchWidth,searchWidth+1):
                        if (0 <= l+k) and (l+k < imgHeight):
                            for m in range(-searchWidth,searchWidth+1):
                                if (0 <= j+m) and (j+m < imgWidth):
                                    if (0 < maskImgD[l+k][j+m]):
                                        if (abs(depthImg[l+k][j+m]-depthImg[k][j]) < threshold):
                                            dbg.add((abs(depthImg[l+k][j+m]-depthImg[k][j]), k, j))
                                            found = True
                                            break
                            if found:
                                break;
                    if found:
                        contPN[k][j] = 255;
    contPO = _ci(searchWidth, threshold, imgHeight, imgWidth, dilImgS, maskImgS, maskImgO, contPO)
    contPS = _ci(searchWidth, threshold, imgHeight, imgWidth, dilImgO, maskImgO, maskImgS, contPS)
    return contPS, contPO

def contact(searchWidth, threshold, imgHeight, imgWidth, s, o, dilatedImgs, maskImgs, depthImg, erodeMasks=True, beSlow=False):
    """
Compute contact masks of two objects. Each object is represented by a pair of masks, a normal and a dilated mask.
The masks are images, all of the same size.
    
Contact is taken to occur where both of the following conditions happen:
    1) there is some overlap between the dilated mask of an object A and the normal mask of the other object B AND
    2) some point on object B that is in that overlap is close to a point on object A.
        
Depth information, represented as a depth image, is used when computing closeness. The depth image has the same
size as the masks.
        
Therefore, a point on an object corresponds to a pixel on its normal mask and a pixel in the depth image. When
searching for points close to a point P, the search will look at pixels in the depth map close to the pixel 
corresponding to P.

Note that when computing closeness, only depth values are compared, rather than a full computation of distance 
between inverse projected pixels.
    
Arguments:
    searchWidth: int, controls the size of the window around a pixel to search for close pixels. The size of the
                 window is then (2*searchWidth+1)^2
    threshold: float, a threshold used to decide whether two points are close or not
    imgHeight, imgWidth: int, gives the masks and depth image dimensions
    s, o: str, name of the objects to look for contacts
    dilatedImgs, maskImgs: dict, where keys are strings and values are numpy arrays of type numpy.uint8. These
                 dictionaries should contain keys equal to string s and o
    depthImg: numpy array of type numpy.float32
    erodeMasks: bool, if True then the normal masks will be slightly eroded before searching for contacts.
    beSlow: bool. Should be kept False in most circumstances. If True, contact masks will be computed using
                 a slower implementation in python. This is/was only useful to debug the C++ implementation
                 by comparing its output to a reference.
    
Returns:
    contPS, contPO: numpy array of type numpy.uint8, contact masks for objects s and o respectively. Will have
                 the same size as the mask images. A non-zero value indicates a pixel was flagged as belonging
                 to the contact region.
    """
    def _pixelCount(mask):
        return len(mask[mask>0])
    def _expand(imgHeight, imgWidth, mask, contact):
        pixelCount = 0.2*_pixelCount(mask)
        contactCount = _pixelCount(contact)
        if 0 == contactCount:
            return contact
        while contactCount < pixelCount:
            contact = numpy.bitwise_and(mask, cv.dilate(contact, dilationKernel))
            contactCount = _pixelCount(contact)
        return contact
    contPS = numpy.zeros((imgHeight, imgWidth),dtype=numpy.uint8)
    contPO = numpy.zeros((imgHeight, imgWidth),dtype=numpy.uint8)
    maskImgS = maskImgs[s]
    maskImgO = maskImgs[o]
    if erodeMasks:
        maskImgS = cv.erode(maskImgs[s], dilationKernel)
        maskImgO = cv.erode(maskImgs[o], dilationKernel)
    if not beSlow:
        _contact(searchWidth, threshold, imgHeight, imgWidth, dilatedImgs[s], dilatedImgs[o], maskImgs[s], maskImgs[o], depthImg, contPS, contPO)
    else:
        contPS, contPO = _contactSlow(searchWidth, threshold, imgHeight, imgWidth, (dilatedImgs[s]), (dilatedImgs[o]), maskImgS, maskImgO, (depthImg), (contPS), (contPO))
    contPS = _expand(imgHeight, imgWidth, maskImgs[s], contPS)
    contPO = _expand(imgHeight, imgWidth, maskImgs[o], contPO)
    return contPS, contPO

