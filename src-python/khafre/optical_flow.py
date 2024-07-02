import cv2 as cv
from khafre.bricks import RatedSimpleQueue, ReifiedProcess
from multiprocessing import Queue
import numpy

def inverseProjection(point, depthImg):
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

class OpticalFlow(ReifiedProcess):
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
        self._goals = RatedSimpleQueue()
        self._imageResults = {}
        self._depthResults = {}
        self._maskResults = {}
        self._currentGoals = []
        self._rateImage = None
        self._droppedImage = 0
        self._rateDepth = None
        self._droppedDepth = 0
        self._rateMask = None
        self._droppedMask = 0
        self._previousImage = None
        self._previousDepth = None
        self._previousMask = None
        self._currentImage = None
        self._currentDepth = None
        self._currentMask = None
        self._previousFeatures = None
        self._currentFeatures = None
        self._featureParams = {}
    def getGoalQueue(self):
        return self._goals
    def _checkSubscriptionRequest(self, name, queue, consumerSHM):
        return name in {"InpImg", "MaskImg", "DepthImg"}
    def _checkPublisherRequest(self, name, queues, consumerSHM):
        return name in {"OutImg", "DbgImg"}
    def doWork(self):
        """
        """
        def _orderQuery(q):
            if q[0]>q[1]:
                return (q[1], q[0])
            return (q[0], q[1])
        haveNew = False
        if not self._goals.empty():
            self._currentGoals, _, _ = self._goals.getWithRates()
        if not self._subscriptions["InpImg"].empty():
            haveNew = True
            self._imageResults, self._rateImage, self._droppedImage = self._subscriptions["InpImg"].getWithRates()
        if not self._subscriptions["DepthImg"].empty():
            haveNew = True
            self._depthResults, self._rateDepth, self._droppedDepth = self._subscriptions["DepthImg"].getWithRates()
        if not self._subscriptions["MaskImg"].empty():
            haveNew = True
            self._maskResults, self._rateMask, self._droppedMask = self._subscriptions["MaskImg"].getWithRates()
        if haveNew and (self._imageResults.get("imgId") == self._depthResults.get("imgId")) and (self._maskResults.get("imgId") == self._depthResults.get("imgId")):
            self._previousImage = self._currentImage
            self._previousDepth = self._currentDepth
            self._previousMask = self._currentMask
            self._previousFeatures = self._currentFeatures
            with self._subscriptions["inpImg"] as inpImg:
                self._currentImage = numpy.uint8(cv.cvtColor(inpImg, cv.COLOR_BGR2GRAY)*255)
            with self._subscriptions["DepthImg"] as depthImg:
                self._currentDepth = numpy.copy(depthImg)
            with self._subscriptions["MaskImg"] as maskImg:
                self._currentMask = numpy.copy(maskImg)
            if (self._currentImage is None) or (self._previousImage is None) or (self._currentMask is None) or (self._previousMask is None) or (self._currentDepth is None) or (self._previousDepth is None):
                return
            featureParams = dict(maxCorners = 5, qualityLevel = 0.2, minDistance = 2, blockSize = 5)
            lkParams = dict(winSize = (15,15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
            queries = set()
            queriesForAll = set()
            for p,s,o in self._currentGoals:
                if "opticalFlowCorners" == p:
                    featureParams["maxCorners"] = int(s)
                elif "opticalFlowQualityLevel" == p:
                    featureParams["qualityLevel"] = int(s)
                elif "opticalFlowMinDistance" == p:
                    featureParams["minDistance"] = int(s)
                elif "opticalFlowBlockSize" == p:
                    featureParams["blockSize"] = int(s)
                elif "opticalFlowMaxLevel" == p:
                    lkParams["maxLevel"] = int(s)
                elif "opticalFlowWinSize" == p:
                    lkParams["winSize"] = (int(s), int(s))
                elif "opticalFlowQuery" == p:
                    if s==o:
                        continue
                    if o is not None:
                        queries.add(_orderQuery((s,o)))
                    else:
                        queriesForAll.add(s)
            qobjs = set([x[0] for x in queries]).union([x[1] for x in queries])
            for x in queriesForAll:
                queries = queries.union([_orderQuery((x, y)) for y in qobjs if x!=y])
            for k, (s,o) in enumerate(queries):
                
            #####if all([x is not None for x in [self._previousImage, self._previousFeatures, self._previousDepth, self._previousMask, self._currentImage, self._currentFeatures, self._currentDepth, self._currentMask]]):
                regionOfInterest = numpy.zeros(self._previousMask.shape, dtype=uint8)
                regionOfInterest[self._previousMask != 0] = 255
                self._currentFeatures = getFeatures(self._previousFeatures, self._previousImage, regionOfInterest, featureParams)
                next, status, error = cv.calcOpticalFlowPyrLK(previousGray, gray, previousFeatures, None, **lk_params)
    def cleanup(self):
        pass



def computeOpticalFlow(previousFeatures, previousGray, gray, maskImg, depthImgPrev, depthImg):
    def _insideMask(x, maskImg):
        cx, cy = x.ravel().astype(int)
        cx = min(imgWidth-1,max(0,cx))
        cy = min(imgHeight-1,max(0,cy))
        return 0 < maskImg[cy][cx]
    if (previousFeatures is None) or (0 == len(previousFeatures)):
        return None, None, None, None
    if maskImg is None:
        previous3D = [inverseProjection(x, depthImgPrev) for x in previousFeatures]
        return previousFeatures, None, previous3D, None
    next, status, error = cv.calcOpticalFlowPyrLK(previousGray, gray, previousFeatures, None, **lk_params)
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
    previous3D = [inverseProjection(x, depthImgPrev) for x in good_old]
    now3D = [inverseProjection(x, depthImg) for x in good_new]
    return good_old, good_new, previous3D, now3D

def getRobotRelativeKinematics(previous3D, now3D):
    if 0 == min(len(now3D),len(previous3D)):
        return None, None
    retq = [(nw - pr) for pr, nw in zip(previous3D, now3D)]
    return sum(now3D)/len(now3D), sum(retq)/len(retq)

def getRelativeMovements(previous3D, now3D, queries):
    def _relativeSpeed(vs, vo, ps, po):
        if (vs is None) or (vo is None) or (ps is None) or (po is None):
            return None
        dVVec = [a-b for a,b in zip(vo, vs)]
        dPVec = [a-b for a,b in zip(po, ps)]
        dPNorm = math.sqrt(sum(x*x for x in dPVec))
        dPDir = [x/dPNorm for x in dPVec]
        return sum(a*b for a,b in zip(dPDir, dVVec))
    kinematicData = {k: getRobotRelativeKinematics(previous3D[k], now3D[k]) for k in now3D.keys() if (previous3D.get(k) is not None) and (now3D[k] is not None)}
    kinematicData["Agent"] = (numpy.array([0,0,0]), numpy.array([0,0,0]))
    retq = []
    for s,oq in queries:
        if oq is None:
            oq = [x for x in kinematicData.keys() if s!=o]
        else:
            oq = [oq]
        for o in oq:
            if (s in kinematicData) and (o in kinematicData):
                dV = _relativeSpeed(kinematicData[s][1], kinematicData[o][1], kinematicData[s][0], kinematicData[o][0])
                if dV is None:
                    continue
                if APPROACHVELOCITY >= dV:
                    retq.append(("approaches", s, o))
                elif DEPARTUREVELOCITY <= dV:
                    retq.append(("departs", s, o))
                else:
                    retq.append(("stillness", s, o))
    return retq

def perception(dbg = False):
    def _objPolygons(label,img,outfile):
        contours, hierarchy = cv.findContours(image=img, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
        contours = [x.reshape((len(x), 2)) for x in contours]
        for polygon, h in zip(contours, hierarchy[0]):
            if 0 > h[3]:
                pstr = ""
                for p in polygon:
                    pstr += ("%f %f " % (p[0]/240., p[1]/240.))
                if 0 < len(polygon):
                    pstr += ("%f %f " % (polygon[0][0]/240., polygon[0][1]/240.))
                _ = outfile.write("%d %s\n" % (label, pstr))    
    perceptionQueriesLocal = {"relativeMovements": [], "contacts": []}
    previousGray, gray = None, None
    previousMaskImgs, maskImgs, previousFeatures, nowFeatures, previous3D, now3D = {}, {}, {}, {}, {}, {}
    maskImgs = {}
    while perceptionGVar.get("keepOn"):
        with imageReady:
            imageReady.wait()
        recognitionResultsLock.acquire()
        image = objectRecognitionResults["image"]
        resultDicts = objectRecognitionResults["resultDicts"]
        depthImg = objectRecognitionResults["depthImg"]
        depthImgPrev = objectRecognitionResults["depthImgPrev"]
        recognitionResultsLock.release()
        resultDicts = remapResultNames(resultDicts)
        if image is None:
            continue
        npImg = numpy.ascontiguousarray(numpy.float32(numpy.array(image)[:,:,(2,1,0)]))/255.0
        previousGray = gray
        gray = numpy.uint8(cv.cvtColor(npImg, cv.COLOR_BGR2GRAY)*255)
        previous3D, now3D = {}, {}
        if (previousGray is None) or (gray is None) or (depthImgPrev is None) or (depthImg is None):
            previousMaskImgs, maskImgs, previousFeatures, nowFeatures = {}, {}, {}, {}
            continue
        perceptionQuestionsLock.acquire()
        perceptionQueriesLocal = {k:v for k,v in perceptionQueries.items()}
        perceptionQuestionsLock.release()
        queriedObjects = getQueriedObjects(perceptionQueriesLocal, resultDicts)
        updatePNDPair(previousMaskImgs, maskImgs, queriedObjects, {e["type"]: e["mask"] for e in resultDicts})
        maskImgs["Agent"] = getSelfMask()
        updatePNDPair(previousFeatures, nowFeatures, queriedObjects, None)
        for o in queriedObjects:
            previousFeatures[o] = getFeatures(previousFeatures.get(o), previousGray, previousMaskImgs.get(o))
            previousFeatures[o], nowFeatures[o], previous3D[o], now3D[o] = computeOpticalFlow(previousFeatures.get(o), previousGray, gray, maskImgs.get(o), depthImgPrev, depthImg)
        relativeMovements = getRelativeMovements(previous3D, now3D, perceptionQueriesLocal["relativeMovements"])
        contacts, contactPixels, contactMasks = getContacts(maskImgs, depthImg, perceptionQueriesLocal["contacts"])
        perceptionResultsLock.acquire()
        perceptionResults["relativeMovements"] = relativeMovements
        perceptionResults["contacts"] = contacts
        perceptionResults["image"] = image
        perceptionResults["contactMasks"] = contactMasks
        perceptionResultsLock.release()
        if dbg:
            #print("QC\n", perceptionQueriesLocal["contacts"])
            print("Relative Movements", sorted(perceptionResults["relativeMovements"]))
            print("Contacts", sorted(perceptionResults["contacts"]))
        with perceptionReady:
            perceptionReady.notify_all()
        if dbg:
            for cd, pixels in contactPixels.items():
                if 0 < len(pixels):
                    print(cd)
                for p in pixels:
                    npImg = cv.line(npImg, p, p, contactColor, 1)
            for k in nowFeatures.keys():
                if (previousFeatures.get(k) is None) or (nowFeatures.get(k) is None):
                    continue
                for i, (new, old) in enumerate(zip(nowFeatures[k], previousFeatures[k])):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    npImg = cv.line(npImg, (a,b), (c,d), opticalFlowColor, 2)
            debugDataLock.acquire()
            debugData["opticalFlow"] = npImg
            debugDataLock.release()


