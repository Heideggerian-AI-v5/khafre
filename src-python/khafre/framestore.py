import itertools
import numpy
import os
import time

import cv2 as cv

from PIL import Image

from khafre.bricks import ReifiedProcess, _Wire

def differentEnough(imageA, imageB, dt, at):
    diff = abs(imageA.astype(numpy.int16)-imageB.astype(numpy.int16))
    diff[diff<dt] = 0
    d = {k:v for k, v in zip(*numpy.unique(diff, return_counts=True))}
    total = sum(v for v in d.values())
    if 0 == total:
        return False
    different = sum(v for k, v in d.items() if 0 != k)
    return (at <= (different/total))

class YOLOFrameSaver(ReifiedProcess):
    def __init__(self):
        super().__init__()
        self._basePath = os.getcwd().replace("\\", "/")
        self._oldImage = None
        self._dbgImg = None
        self._dt = 5
        self._at = 0.1
    def _checkPublisherRequest(self, name: str, wire: _Wire):
        return ("DbgImg" == name)
    def _checkSubscriptionRequest(self, name: str, wire: _Wire):
        return ("InpImg" == name)
    def _handleCommand(self, command):
        op, args = command
        if "SET_PATH" == op:
            self._basePath = args[0]
        elif "SET_DIFFERENCE_THRESHOLD" == op:
            self._dt = args[0]
        elif "SET_AMOUNT_THRESHOLD" == op:
            self._at = args[0]
    def _doWork(self):
        def _set2str(s):
            return '_'.join(sorted([str(x) for x in s]))
        annotation = self._dataFromSubscriptions["InpImg"]["notification"]
        image = self._dataFromSubscriptions["InpImg"]["image"]
        height, width, channels = image.shape
        if (0 < len(annotation)) and ((self._oldImage is None) or (differentEnough(image, self._oldImage, self._dt, self._at))):
            if self.havePublisher("DbgImg"):
                if self._dbgImg is None:
                    self._dbgImg = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.float32)
                numpy.copyto(self._dbgImg, image / 255.0) 
            self._oldImage = image
            fnamePrefix = os.path.join(self._basePath, "seg_%s" % str(time.perf_counter()))
            imageBGR = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            Image.fromarray(imageBGR).save(fnamePrefix + ".jpg")
            with open(fnamePrefix + ".txt", "w") as outfile:
                for desc in annotation:
                    polygons, semantics = desc["polygons"], desc["semantics"]
                    print(polygons)
                    label = "partOf_%s_usedFor_%s_asRole_%s" % (_set2str(semantics.get("masksPartOfObjectType", [])), _set2str(semantics.get("usedForTaskType", [])), _set2str(semantics.get("playsRoleType", [])))
                    if self.havePublisher("DbgImg"):
                        labelHash = hash(label)
                        color = (((labelHash&0xFF0000) >> 16)/255.0, ((labelHash&0xFF00) >> 8)/255.0, (labelHash&0xFF)/255.0)
                    for polygon in polygons:
                        if self.havePublisher("DbgImg"):
                            cv.fillPoly(self._dbgImg, pts = [polygon], color = color)
                        pstr = ""
                        for p in polygon:
                            pstr += ("%f %f " % (p[0]/width, p[1]/height))
                        if 0 < len(polygon):
                            pstr += ("%f %f " % (polygon[0][0]/width, polygon[0][1]/height))
                        _ = outfile.write("%s %s\n" % (label, pstr))
            if self.havePublisher("DbgImg"):
                self._requestToPublish("DbgImg", "Storage", self._dbgImg)


