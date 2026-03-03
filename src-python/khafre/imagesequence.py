import ctypes
import cv2 as cv
from khafre.imagesource import ImageSource
from multiprocessing import Value
import mimetypes
import numpy
import os
import pickle
from PIL import Image
import socket
import struct
import time

def imgDimPicker(inputImageFolder):
    imgHeight = 0
    imgWidth = 0
    for x in os.listdir(inputImageFolder):
        if (mimetypes.guess_type(x)[0] is not None) and (mimetypes.guess_type(x)[0].startswith("image/")):
            newHeight, newWidth = numpy.asarray(Image.open(os.path.join(inputImageFolder, x))).shape[:2]
            if newHeight > imgHeight:
                imgHeight = newHeight
            if newWidth > imgWidth:
                imgWidth = newWidth
    #return imgHeight, imgWidth
    return 480, 640

class ImageSequence(ImageSource):
    def __init__(self):
        super().__init__()
        self._filePaths = []
        self._crFileIdx = 0
        self._ended = Value(ctypes.c_int8)
        self._ended.value = 1
    def hasEnded(self):
        return (0 != self._ended.value)
    def waitForLoad(self):
        while (0 != self._ended.value):
            time.sleep(0.1)
    def waitForEnd(self):
        while (0 == self._ended.value):
            time.sleep(0.1)
    def _handleCommand(self, command):
        def _maybeImage(name):
            gt = mimetypes.guess_type(name)[0]
            if gt is not None:
                return gt.startswith("image/")
            return False
        op, args = command
        if "LOAD" == op:
            self._crFileIndex = 0
            self._filePaths = [os.path.join(args[0], x) for x in os.listdir(args[0]) if _maybeImage(x)]
            print("Loaded images from %s" % args[0])
            self._ended.value = 0
    def _doWork(self):
        if ([] == self._filePaths) or (self._crFileIdx >= len(self._filePaths)) or (0 != self._ended.value):
            self._ended.value = 1
            return
        c = time.perf_counter()
        fileName = self._filePaths[self._crFileIdx]
        image = numpy.asarray(Image.open(fileName))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self._crFileIdx += 1
        idData = {"imgId": os.path.split(fileName)[-1]}
        print("SENDING image", idData["imgId"])
        self._requestToPublish("OutImg", idData, image)
        if self.havePublisher("DbgImg"):
            dbgImg = image.astype(numpy.float32)/255.0
            self._requestToPublish("DbgImg", str(idData), dbgImg)
