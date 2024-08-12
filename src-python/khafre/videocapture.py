import ctypes
import cv2 as cv
from khafre.imagesource import ImageSource
from multiprocessing import Value
import numpy
import time

class RecordedVideoFeed(ImageSource):
    def __init__(self):
        super().__init__()
        self._dbgImg = None
        self._bypassEvent = False
        self._videoCapture = None
        self._atVideoT = 0
        self._atRealT = None
        self._ended = Value(ctypes.c_int8)
        self._ended.value = 1
    def hasEnded(self):
        return (0 != self._ended.value)
    def _handleCommand(self, command):
        op, args = command
        if "LOAD" == op:
            self._videoCapture = cv.VideoCapture(args[0])
            self._atVideoT = 0
            self._atRealT = None
            self._ended.value = 0
    def _doWork(self):
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
    def _cleanup(self):
        pass
