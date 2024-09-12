import ctypes
import cv2 as cv
from khafre.imagesource import ImageSource
from multiprocessing import Value
import numpy
import pickle
import socket
import struct
import time

class RecordedVideoFeed(ImageSource):
    def __init__(self):
        super().__init__()
        self._bypassEvent = False
        self._videoCapture = None
        self._atVideoT = 0
        self._atRealT = None
        self._ended = Value(ctypes.c_int8)
        self._ended.value = 1
        self.atFrame = 0
    def hasEnded(self):
        return (0 != self._ended.value)
    def _handleCommand(self, command):
        op, args = command
        if "LOAD" == op:
            self._videoCapture = cv.VideoCapture(args[0])
            print("Loaded video %s" % args[0])
            self._atVideoT = 0
            self._atRealT = None
            self._ended.value = 0
    def _doWork(self):
        if (self._videoCapture is None) or (not self._videoCapture.isOpened()) or (0 != self._ended.value):
            return
        if 1000 >= self.atFrame:
            #time.sleep(60)
            self._atVideoT = 0
            self._atRealT = None
        self.atFrame += 1
        c = time.perf_counter()
        frameT = 1.0/self._videoCapture.get(cv.CAP_PROP_FPS)
        if self._atRealT is not None:
            self._atVideoT = round(self._atVideoT + max((c - self._atRealT), frameT), 2)
        self._atRealT = c
        self._videoCapture.set(cv.CAP_PROP_POS_MSEC,self._atVideoT*1000)
        hasFrames, image = self._videoCapture.read()
        if hasFrames:
            idData = {"imgId": str(time.perf_counter())}
            self._requestToPublish("OutImg", idData, image)
            if self.havePublisher("DbgImg"):
                if self._dbgImg is None:
                    self._dbgImg = numpy.zeros(image.shape, numpy.float32)
                self._dbgImg = image.astype(numpy.float32)/255.0
                self._requestToPublish("DbgImg", str(idData), self._dbgImg)
        else:
            self._ended.value = 1

class WebStream(ImageSource):
    def __init__(self):
        super().__init__()
        self._videoCapture = None
        self._payloadSize = struct.calcsize("Q")
    def _handleCommand(self, command):
        op, args = command
        if "CONNECT" == op:
            uri = args[0]
            if self._videoCapture is not None:
                self._videoCapture.release()
            self._videoCapture = cv.VideoCapture(uri)
    def _doWork(self):
        if (self._videoCapture is not None) and (self._videoCapture.isOpened()):
            ret, image = self._videoCapture.read()
            if ret:
                idData = {"imgId": str(time.perf_counter())}
                self._requestToPublish("OutImg", idData, image)
                if self.havePublisher("DbgImg"):
                    if self._dbgImg is None:
                        self._dbgImg = numpy.zeros(image.shape, numpy.float32)
                    self._dbgImg = image.astype(numpy.float32)/255.0
                    self._requestToPublish("DbgImg", str(idData), self._dbgImg)
            else:
                time.sleep(0.01)
        else:
            time.sleep(0.01)
    def _cleanup(self):
        if self._videoCapture is not None:
            self._videoCapture.release()
