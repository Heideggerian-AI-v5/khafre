import itertools
import numpy
import os
import time

import cv2 as cv

from PIL import Image

from khafre.bricks import ReifiedProcess, _Wire

class DownPointer(ReifiedProcess):
    def __init__(self):
        super().__init__()
    def _checkPublisherRequest(self, name: str, wire: _Wire):
        return ("DownDir" == name)
    def _checkSubscriptionRequest(self, name: str, wire: _Wire):
        return False
    def _getDown(self):
        raise NotImplementedError
    def _doWork(self):
        if self.havePublisher("DownDir"):
            self._requestToPublish("DownDir", {"down": self._getDown()}, None)

class ConstantDownPointer(DownPointer):
    def __init__(self):
        super().__init__()
    def _getDown(self):
        return [0,1,0]
