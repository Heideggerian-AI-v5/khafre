import cv2 as cv
from khafre.bricks import ReifiedProcess
from multiprocessing import Queue
import numpy
import time

'''
import cv2
vidcap = cv2.VideoCapture('video.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 0.5 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
	success = getFrame(sec)
'''

class RecordedVideoFeed(ReifiedProcess):
    def __init__(self):
        super().__init__()
        self._videoCapture = None
        self._atVideoT = 0
        self._atRealT = None
        self._command = Queue()
        self._ended = Queue()
    def _checkPublisherRequest(self, name, queues, consumerSHM):
        return name in {"OutImg", "DbgImg"}
    def _checkSubscriptionRequest(self, name, queue, consumerSHM):
        return False
    def sendCommand(self, command, block=False, timeout=None):
        self._command.put(command, block=block, timeout=timeout)
    def hasEnded(self):
        if not self._ended.empty():
            self._ended.get()
            return True
        return False
    def _handleCommand(self, command):
        op, args = command
        if "LOAD" == op:
            self._videoCapture = cv.VideoCapture(args[0])
            self._atVideoT = 0
            self._atRealT = None
        elif "FRAME":
            c = time.perf_counter()
            frameT = 1.0/self._videoCapture.get(cv.CAP_PROP_FPS)
            if self._atRealT is not None:
                self._atVideoT = round(self._atVideoT + max((c - self._atRealT), frameT), 2)
            self._atRealT = c
            self._videoCapture.set(cv.CAP_PROP_POS_MSEC,self._atVideoT*1000)
            hasFrames, image = self._videoCapture.read()
            if hasFrames:
                image = image.astype(numpy.float32)/255.0
                for name in {"OutImg", "DbgImg"}:
                    if self._publishers.get(name):
                        self._publishers[name].publish(image, True)
            else:
                self._ended.put(True)
    def doWork(self):
        while not self._command.empty():
            self._handleCommand(self._command.get())
    def cleanup(self):
        pass