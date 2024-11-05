import cv2 as cv
from pycpd import AffineRegistration
import numpy

from khafre.polygons import findTopPolygons
from khafre.taskable import TaskableProcess

'''
y = x.R+t
x = (y-t).Rinv
'''

class Decluder(Taskable):
    def __init__(self):
        super().__init__()
        self._prefix="declude"
        self._settings["maxAge"] = 30
        self._settings["minOverlap"] = 100
        self._polygonData = {}
    def _checkSubscriptionRequest(self, name, wire):
        return name in {"InpImg", "MaskImg"}
    def _checkPublisherRequest(self, name, wire):
        return name in {"OutImg", "DbgImg"}
    def _performStep(self):
        """
        """
        # TODO: use depthImg for estimating when an object is not occluded and estimated shape may be trimmed, rather than validPixels
        # Read inputs
        inpResults, inpImg, rateInp, droppedInp = self._requestSubscribedData("InpImg")
        maskResults, maskImg, rateMask, droppedMask = self._requestSubscribedData("MaskImg")
        masks = {s["name"]: {"polygons": s["polygons"], "confidence": s["confidence"]} for s in maskResults["segments"]}
        # Prepare queries
        qUniverse = [x["name"] for x in maskResults.get("segments", [])]
        self._fillInGenericQueries(qUniverse)
        sObjs = set([x[1] for x in self._queries])
        qobjs = sObjs.union([x[2] for x in self._queries])
        relevantOverlaps = {}
        for q in self._queries:
            _, s, o = q
            if s not in relevantOverlaps:
                relevantOverlaps[s] = set()
            relevantOverlaps[s].add(o)
        
        # Make a mask for all pixels believed to belong to some recognizable object
        validPixels = numpy.zeros(maskImg.shape[:2], dtype=numpy.uint8)
        for name, maskData in masks.items():
            if name in qObjs:
                maskData["maskIst"] = numpy.zeros(maskImg.shape[:2], dtype=numpy.uint8)
                for p in maskData["polygons"]:
                    cv.fillPoly(maskData["maskIst"], pts = [p], color = 255)
                validPixels[maskData["maskIst"] != 0] = 255
            else:
                for p in maskData["polygons"]:
                    cv.fillPoly(validPixels, pts = [p], color = 255)

        # Remove irrelevant entries from self._polygonData:
        #     objects we are no longer interested in tracking overlaps for
        for name in set(self._polygonData.keys()).difference(sObjs):
            self._polygonData.pop(name)
        #     and overlaps we no longer care about
        for name, data in self._polygonData.items():
            for e in set(self._polygonData[name]["previousOverlaps"].keys()).difference(relevantOverlaps[name]):
                self._polygonData[name]["previousOverlaps"].pop(e)

        # Update polygons
        scratchpad = numpy.zeros(maskImg.shape[:2], dtype=numpy.uint8)
        for name, maskData in masks.items():
            if name not in sObjs:
                continue
            if (name not in self._polygonData):
                self._polygonData[name] = {"polygons": maskData["polygons"], "previousOverlaps": {}}
            else:
                polyTarget = numpy.concatenate(maskData["polygons"], direction=0)
                polyStart = numpy.concatenate(self._polygonData[name]["polygons"], direction=0)
                reg = AffineRegistration(**{'X': polyTarget, 'Y': polyStart})
                reg.register(lambda iteration=0, error=0, X=None, Y=None : None)
                Re, te = reg.get_registration_parameters()
                tPolygons = [numpy.dot(x, Re) + te for x in self._polygonData[name]["polygons"]]
                for p in tPolygons:
                    cv.fillPoly(scratchpad, pts = [p], color = 255)
                scratchPad[maskData["maskIst"] != 0] = 255
                scratchPad[validPixels == 0] = 0
                self._polygonData[name]["polygons"] = findTopPolygons(scratchpad)
                maskData["maskSoll"] = numpy.copy(scratchpad)
                scratchPad[:,:] = 0
            self._polygonData[name]["age"] = -1

        declusionMasks = []
        triples = set()
        maskId = 0
        occImg = numpy.zeros((maskImg.shape[0], maskImg.shape[1]), dtype=numpy.uint32)
        for k, (s, p, o) in enumerate(self._queries):
            # Test for current occlusion of s by o
            scratchpad[masks[s]["maskSoll"] == 0] = 0
            scratchpad[masks[s]["maskSoll"] != 0] = 255
            scratchpad[masks[o]["maskIst"] == 0] = 0
            # scratchpad now stores a bitmap showing the current occlusion of s by o
            cols, counts = numpy.unique(scratchpad, return_counts=True)
            overlap = {k:v for k,v in zip(cols, counts)}[255]
            if self._settings["minOverlap"] <= overlap:
                triples.add(("occludedBy", s, o))
            _ = [cv.fillPoly(scratchpad, pts = [p], color = 255) for p in self._polygonData[s]["previousOverlaps"].get(o, [])]
            # scratchpad now stores a bitmap showing current and past occlusion of s by o
            self._polygonData[s]["previousOverlaps"][o] = findTopPolygons(scratchpad)
            scratchpad[masks[s]["maskIst"] == 0] = 0
            # scratchpad now stores a bitmap showing what was occluded in the past by o but is now visible
            cols, counts = numpy.unique(scratchpad, return_counts=True)
            decluded = {k:v for k,v in zip(cols, counts)}[255]
            if self._settings["minOverlap"] <= decluded:
                maskId += 1
                declusionMasks.append({"hasId": maskId, "hasP": "declusionOf", "hasS": s, "hasO": o, "polygons": findTopPolygons(scratchpad)})
                occImg[scratchpad != 0] = maskId
        self._requestToPublish("OutImg", {"imgId": maskResults.get("imgId"), "masks": masks, "triples": triples}, occImg)

        # Remove polygons that are too old. Renew! Renew!
        names = list(self._polygonData.keys())
        for name in names:
            data = self._polygonData[name]
            data["age"] += 1
            if data["age"] > self._settings["maxAge"]:
                self._polygonData.pop(name)
        
        # Do we need to prepare a debug image?
        if self.havePublisher("DbgImg"):
            if maskImg.shape[:2] == inpImg.shape[:2]:
                dbgImg = inpImage.astype(numpy.float32) / 255
            else:
                dbgImg = cv.resize(inpImage, (maskImg.shape[1], maskImg.shape[0], 3), interpolation=cv.INTER_LINEAR).astype(numpy.float32) / 255)
            for s in sObjs:
                if s in self._polygonData:
                    polygons = self._polygonData[s]["polygons"]
                    h = hash(s)
                    color = (((h&0xFF0000) >> 16) / 255.0, ((h&0xFF00) >> 8) / 255.0, ((h&0xFF)) / 255.0)
                    cv.polylines(dbgImg, polygons, True, color, 3)
                    for o, polygons in self._polygonData.get(s, {"previousOverlaps": {}}):
                        h = hash((s,o))
                        color = (((h&0xFF0000) >> 16) / 255.0, ((h&0xFF00) >> 8) / 255.0, ((h&0xFF)) / 255.0)
                        cv.polylines(dbgImg, polygons, True, color, 3)
            self._requestToPublish("DbgImg","", dbgImg)

