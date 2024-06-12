import ctypes 
import numpy
from numpy.ctypeslib import ndpointer 
import os
import platform
import cv2 as cv

from .contactcpp import _contact

_datapp = ndpointer(dtype=numpy.uintp, ndim=1, flags='C') 


#dirpath = os.path.dirname(os.path.abspath(__file__))
#if 'Windows' == platform.system():
#    soPath = os.path.join(dirpath, 'libcontact.dll')
#else:
#    soPath = os.path.join(dirpath, 'libcontact.so')
#
#_dll = ctypes.CDLL(soPath) 
#
#_contact = _dll.contact
#                      searchWidth     threshold      rows           cols          dilS     dilO      maskS   maskO    depth    contPS   contPO
#_contact.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int, _datapp, _datapp, _datapp, _datapp, _datapp, _datapp, _datapp]
#_contact.restype = None

dilationKernel = cv.getStructuringElement(cv.MORPH_CROSS,(3,3),(1,1))

def _contactSlow(searchWidth, threshold, imgHeight, imgWidth, dilImgS, dilImgO, maskImgS, maskImgO, depthImg, contPS, contPO):
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

def contact(searchWidth, threshold, imgHeight, imgWidth, s, o, dilatedImgs, maskImgs, depthImg):
    def _pixelCount(mask):
        return len(mask[mask>0])
    def _expand(imgHeight, imgWidth, mask, contact):
        #return contact
        pixelCount = 0.2*_pixelCount(mask)
        contactCount = _pixelCount(contact)
        #print("PC", pixelCount, contactCount)
        if 0 == contactCount:
            return contact
        while contactCount < pixelCount:
            contact = numpy.bitwise_and(mask, cv.dilate(contact, dilationKernel))
            contactCount = _pixelCount(contact)
            #print("    ", pixelCount, contactCount)
        return contact
    def _pp(x):
        return (x.__array_interface__['data'][0] + numpy.arange(x.shape[0])*x.strides[0]).astype(numpy.uintp)
    contPS = numpy.zeros((imgHeight, imgWidth),dtype=numpy.uint8)
    contPO = numpy.zeros((imgHeight, imgWidth),dtype=numpy.uint8)
    #searchWidth = 7
    #threshold = 0.5
    maskImgS = maskImgs[s]
    maskImgO = maskImgs[o]
    #maskImgS = cv.erode(maskImgs[s], dilationKernel)
    #maskImgO = cv.erode(maskImgs[o], dilationKernel)
    _contact(searchWidth, threshold, imgHeight, imgWidth, dilatedImgs[s], dilatedImgs[o], maskImgs[s], maskImgs[o], depthImg, contPS, contPO)
    #_contact(searchWidth, threshold, imgHeight, imgWidth, _pp(dilatedImgs[s]), _pp(dilatedImgs[o]), _pp(maskImgs[s]), _pp(maskImgs[o]), _pp(depthImg), _pp(contPS), _pp(contPO))
    #_contact(searchWidth, threshold, imgHeight, imgWidth, _pp(dilatedImgs[s]), _pp(dilatedImgs[o]), _pp(maskImgs[s]), _pp(maskImgs[o]), _pp(depthImg), _pp(contPS), _pp(contPO))
    #contPS, contPO = _contactSlow(searchWidth, threshold, imgHeight, imgWidth, (dilatedImgs[s]), (dilatedImgs[o]), maskImgS, maskImgO, (depthImg), (contPS), (contPO))
    contPS = _expand(imgHeight, imgWidth, maskImgs[s], contPS)
    contPO = _expand(imgHeight, imgWidth, maskImgs[o], contPO)
    return contPS, contPO
