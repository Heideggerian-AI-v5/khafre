import itertools
import os
import time

import cv2 as cv

from PIL import Image

from khafre.bricks import ReifiedProcess, _Wire

def differentEnough(imageA, imageB, dt):
    return True

class YOLOFrameSaver(ReifiedProcess):
    def __init__(self):
        super().__init__()
        self._basePath = os.getcwd().replace("\\", "/")
        self._oldImage = None
        self._dt = None
    def _checkPublisherRequest(self, name: str, wire: _Wire):
        return False
    def _checkSubscriptionRequest(self, name: str, wire: _Wire):
        return ("InpImg" == name)
    def _handleCommand(self, command):
        op, args = command
        if "SET_PATH" == op:
            self._basePath = args[0]
        if "SET_DIFFERENCE_THRESHOLD" == op:
            self._dt = args[0]
    def _doWork(self):
        def _set2str(s):
            return '_'.join(sorted([str(x) for x in s]))
        annotation = self._dataFromSubscriptions["InpImg"]["notification"]
        image = self._dataFromSubscriptions["InpImg"]["image"]
        print("FS", len(annotation))
        height, width, channels = image.shape
        if (0 < len(annotation)) and ((self._oldImage is None) or (differentEnough(image, self._oldImage, self._dt))):
            self._oldImage = image
            fnamePrefix = os.path.join(self._basePath, "seg_%s" % str(time.perf_counter()))
            imageBGR = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            Image.fromarray(imageBGR).save(fnamePrefix + ".jpg")
            print(" ... saved img", fnamePrefix)
            with open(fnamePrefix + ".txt", "w") as outfile:
                for desc in annotation:
                    contours, hierarchy, semantics = desc["contours"], desc["hierarchy"], desc["semantics"]
                    label = "partOf_%s_usedFor_%s_asRole_%s" % (_set2str(semantics.get("masksPartOfObjectType", [])), _set2str(semantics.get("usedForTaskType", [])), _set2str(semantics.get("playsRoleType", [])))
                    for polygon, h in zip(contours, hierarchy[0]):
                        if (0 > h[3]) and (2 < len(polygon)):
                            pstr = ""
                            for p in polygon:
                                pstr += ("%f %f " % (p[0]/width, p[1]/height))
                            if 0 < len(polygon):
                                pstr += ("%f %f " % (polygon[0][0]/width, polygon[0][1]/height))
                            _ = outfile.write("%s %s\n" % (label, pstr))
            print(" ... saved txt")

