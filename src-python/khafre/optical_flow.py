import cv2 as cv
from khafre.bricks import ReifiedProcess
from khafre.taskable import TaskableProcess
import math
import numpy

def inverseProjection(point, depthImg, f):
    if isinstance(point, numpy.ndarray):
        u,v = point.ravel().astype(int)
    else:
        u,v = point
    imgHeight, imgWidth = depthImg.shape
    u = min(imgWidth-1,max(0,u))
    v = min(imgHeight-1,max(0,v))
    depth = depthImg[v][u]
    xi = (u - imgWidth/2.0)*depth/(f)
    yi = (v - imgHeight/2)*depth/(f)
    return numpy.array((xi, yi, depth))

def getFeatures(previousFeatures, previousGray, previousMaskImg, featureParams):
    if previousMaskImg is None:
        return None
    if previousFeatures is None:
        previousFeatures = numpy.array([])
    adjFeatureParams = featureParams.copy()
    alreadyPresent = 0
    alreadyPresent = len(previousFeatures)
    adjFeatureParams['maxCorners'] = featureParams['maxCorners'] - alreadyPresent
    if (0 < adjFeatureParams['maxCorners']):
        newFeatures = cv.goodFeaturesToTrack(previousGray, mask=previousMaskImg, **adjFeatureParams)
        if newFeatures is not None:
            newFeatures = newFeatures.astype(numpy.float32)
            if (0 < len(newFeatures)):
                if (0 < len(previousFeatures)):
                    previousFeatures = numpy.concatenate((previousFeatures, newFeatures))
                else:
                    previousFeatures = newFeatures
    return previousFeatures

def computeOpticalFlow(previousFeatures, previousGray, gray, maskImg, depthImgPrev, depthImg, lkParams, f):
    def _insideMask(x, maskImg):
        cx, cy = x.ravel().astype(int)
        cx = min(maskImg.shape[1]-1,max(0,cx))
        cy = min(maskImg.shape[0]-1,max(0,cy))
        return 0 < maskImg[cy][cx]
    if (previousFeatures is None) or (0 == len(previousFeatures)):
        return [], [], None, None
    next, status, error = cv.calcOpticalFlowPyrLK(previousGray, gray, previousFeatures, None, **lkParams)
    good_old = numpy.array([],dtype=numpy.float32)
    good_new = numpy.array([],dtype=numpy.float32)
    if next is not None:
        good_old = previousFeatures[status == 1]
        good_old = good_old.reshape((len(good_old), 1, 2))
        good_new = next[status==1].astype(numpy.float32)
        good_new = good_new.reshape((len(good_new), 1, 2))
    onSameObj = [_insideMask(x, maskImg) for x in good_new]
    aux = [(x,y) for x,y,z in zip(good_new, good_old, onSameObj) if z]
    good_new = numpy.array([x[0] for x in aux]).reshape((len(aux),1,2)).astype(numpy.float32)
    good_old = numpy.array([x[1] for x in aux]).reshape((len(aux),1,2)).astype(numpy.float32)
    previous3D = [inverseProjection(x, depthImgPrev, f) for x in good_old]
    now3D = [inverseProjection(x, depthImg, f) for x in good_new]
    return good_old, good_new, previous3D, now3D

def getRobotRelativeKinematics(previous3D, now3D):
    if 0 == min(len(now3D),len(previous3D)):
        return None, None
    retq = [(nw - pr) for pr, nw in zip(previous3D, now3D)]
    return sum(now3D)/len(now3D), sum(retq)/len(retq)

def getRelativeMovements(previous3D, now3D, queries, approachV, departV):
    def _relativeSpeed(vs, vo, ps, po):
        if (vs is None) or (vo is None) or (ps is None) or (po is None):
            return None
        dVVec = [a-b for a,b in zip(vo, vs)]
        dPVec = [a-b for a,b in zip(po, ps)]
        dPNorm = max(0.0001, math.sqrt(sum(x*x for x in dPVec)))
        dPDir = [x/dPNorm for x in dPVec]
        return sum(a*b for a,b in zip(dPDir, dVVec))
    kinematicData = {k: getRobotRelativeKinematics(previous3D[k], now3D[k]) for k in now3D.keys() if (previous3D.get(k) is not None) and (now3D[k] is not None)}
    kinematicData["self"] = (numpy.array([0,0,0]), numpy.array([0,0,0]))
    retq = set()
    for p, s, o in queries:
        if (s in kinematicData) and (o in kinematicData):
            dV = _relativeSpeed(kinematicData[s][1], kinematicData[o][1], kinematicData[s][0], kinematicData[o][0])
            if dV is None:
                continue
            if approachV >= dV:
                retq.add(("approaches", s, o))
                retq.add(("approaches", o, s))
            elif departV <= dV:
                retq.add(("departs", s, o))
                retq.add(("departs", o, s))
            else:
                retq.add(("stillness", s, o))
                retq.add(("stillness", o, s))
    return retq

class OpticalFlow(TaskableProcess):
    """
Subprocess in which contact masks are calculated based on an object mask image and a depth image.

Wires supported by this subprocess:
    MaskImg: subscription. The object mask image and associated segmentation results.
    DepthImg: subscription. The depth image and associated notification data.
    OutImg: publisher. The contact masks image and associated symbolic results.
    DbgImg: publisher. An image of the contact masks.

Additionally, gets goal data (sets of triples) from a queue.
    """
    def __init__(self):
        super().__init__()
        self._dbgImg = None
        self._prefix="opticalFlow"
        self._settings["featureParams"] = dict(maxCorners = 10, qualityLevel = 0.2, minDistance = 2, blockSize = 5)
        self._settings["lkParams"] = dict(winSize = (15,15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self._settings["approachVelocity"] = -0.02
        self._settings["departVelocity"] = 0.02
        self._settings["focalDistance"] = 200.0
        self._symmetricPredicates.add("opticalFlow/query/relativeMovement")
        self._previousImage = None
        self._previousDepth = None
        self._previousMask = None
        self._currentImage = None
        self._currentDepth = None
        self._currentMask = None
        self._previousMaskImgs = {}
        self._currentMaskImgs = {}
        self._previousFeatures = {}
    def _checkSubscriptionRequest(self, name, wire):
        return name in {"InpImg", "MaskImg", "DepthImg"}
    def _checkPublisherRequest(self, name, wire):
        return name in {"OutImg", "DbgImg"}
    def _performStep(self):
        """
        """
        def _getMaskImgs(maskImg, results, qObjs):
            results = results.get("segments", [])
            retq = {}
            for e in results:
                if e["type"] in qObjs:
                    retq[e["type"]] = numpy.zeros(maskImg.shape,dtype=numpy.uint8)
                    cv.fillPoly(retq[e["type"]], pts = [e["polygon"]], color = 255)
            return retq
        self._previousImage = self._currentImage
        self._previousDepth = self._currentDepth
        self._previousMask = self._currentMask
        imageResults, inpImage, rateImage, droppedImage = self._requestSubscribedData("InpImg")
        depthResults, self._currentDepth, rateDepth, droppedDepth = self._requestSubscribedData("DepthImg")
        maskResults, self._currentMask, rateMask, droppedMask = self._requestSubscribedData("MaskImg")
        self._currentImage = cv.cvtColor(inpImage, cv.COLOR_BGR2GRAY) #numpy.uint8(cv.cvtColor(inpImg, cv.COLOR_BGR2GRAY)*255)
        qUniverse = [x["type"] for x in maskResults.get("segments", [])]
        self._fillInGenericQueries(qUniverse)
        if not ((self._currentImage is None) or (self._previousImage is None) or (self._currentMask is None) or (self._previousMask is None) or (self._currentDepth is None) or (self._previousDepth is None)):
            featureParams = self._settings["featureParams"]
            lkParams = self._settings["lkParams"]
            f = self._settings["focalDistance"]
            qObjs = [x for x in set([x[1] for x in self._queries]).union([x[2] for x in self._queries])]
            self._previousMaskImgs = self._currentMaskImgs
            self._currentMaskImgs = _getMaskImgs(self._currentMask, maskResults, qObjs)
            qObjs = [x for x in qObjs if (x in self._currentMaskImgs) and (x in self._previousMaskImgs)]
            _=[self._previousFeatures.pop(x) for x in set(self._previousFeatures.keys()).difference(qObjs)]
            nowFeatures={}
            previous3D={}
            now3D={}
            for o in qObjs:
                self._previousFeatures[o] = getFeatures(self._previousFeatures.get(o), self._previousImage, self._previousMaskImgs[o], featureParams)
                self._previousFeatures[o], nowFeatures[o], previous3D[o], now3D[o] = computeOpticalFlow(self._previousFeatures.get(o), self._previousImage, self._currentImage, self._currentMaskImgs[o], self._previousDepth, self._currentDepth, lkParams, f)
            relativeMovements = getRelativeMovements(previous3D, now3D, self._queries, self._settings["approachVelocity"], self._settings["departVelocity"])
            self._requestToPublish("OutImg", {"imgId": maskResults.get("imgId"), "triples": relativeMovements}, None)
            # Do we need to prepare a debug image?
            if self.havePublisher("DbgImg"):
                if self._dbgImg is None:
                    self._dbgImg = numpy.zeros((self._currentImage.shape[0], self._currentImage.shape[1], 3), numpy.float32)
                numpy.copyto(self._dbgImg, cv.cvtColor(self._currentImage.astype(self._dbgImg.dtype) / 255, cv.COLOR_GRAY2BGR))
                for k in set(nowFeatures.keys()).intersection(self._previousFeatures.keys()):
                    for i, (new, old) in enumerate(zip(nowFeatures[k], self._previousFeatures[k])):
                        a, b = new.ravel().astype(int)
                        c, d = old.ravel().astype(int)
                        self._dbgImg = cv.line(self._dbgImg, (a,b), (c,d), (1.0,0.5,1.0), 1)
                        self._dbgImg = cv.rectangle(self._dbgImg,(c-2,d-2),(c+2,d+2),(1.0,0.5,1.0),-1)
                self._requestToPublish("DbgImg","%.02f %.02f ifps | %d%% %d%% obj drop" % (rateMask if rateMask is not None else 0.0, rateDepth if rateDepth is not None else 0.0, droppedMask, droppedDepth), self._dbgImg)
            self._previousFeatures = nowFeatures
        elif not ((self._currentImage is None) or (self._currentMask is None) or (self._currentDepth is None)):
            self._requestToPublish("OutImg", {"imgId": 0, "triples": []}, None)
    def _cleanup(self):
        pass
