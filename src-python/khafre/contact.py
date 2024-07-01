import ctypes 
import cv2 as cv
from khafre.bricks import RatedSimpleQueue, ReifiedProcess
from multiprocessing import Queue
import numpy
from numpy.ctypeslib import ndpointer 
import os
import platform

from .contactcpp import _contact


dilationKernel = cv.getStructuringElement(cv.MORPH_CROSS,(3,3),(1,1))

def _contactSlow(searchWidth, threshold, imgHeight, imgWidth, dilImgS, dilImgO, maskImgS, maskImgO, depthImg, contPS, contPO):
    """
Should not call this. A naive implementation of searching for contact masks in python. Too slow for practical use,
present here to have a kind of easier to check reference and tester for the faster implementation in C++.
    """
    def _ci(searchWidth, threshold, imgHeight, imgWidth, dilImgD, maskImgD, maskImgN, contPN):
        #dKS = cv.getStructuringElement(cv.MORPH_RECT,(searchWidth*2+1,searchWidth*2+1),(searchWidth,searchWidth))
        #contPN = numpy.bitwise_and(dilImgD,maskImgN)
        contPN = numpy.array(maskImgN)
        return contPN
        dbg = set()
        for k in range(imgHeight):
            for j in range(imgWidth):
                if (0 < maskImgN[k][j]) and (0 < dilImgD[k][j]):
                    found = False
                    for l in range(-searchWidth,searchWidth+1):
                        if (0 <= l+k) and (l+k < imgHeight):
                            for m in range(-searchWidth,searchWidth+1):
                                if (0 <= j+m) and (j+m < imgWidth):
                                    if (0 < maskImgD[l+k][j+m]):
                                        if (abs(depthImg[l+k][j+m]-depthImg[k][j]) < threshold):
                                            dbg.add((abs(depthImg[l+k][j+m]-depthImg[k][j]), k, j))
                                            found = True
                                            break
                            if found:
                                break;
                    if found:
                        contPN[k][j] = 255;
    contPO = _ci(searchWidth, threshold, imgHeight, imgWidth, dilImgS, maskImgS, maskImgO, contPO)
    contPS = _ci(searchWidth, threshold, imgHeight, imgWidth, dilImgO, maskImgO, maskImgS, contPS)
    return contPS, contPO

def contact(searchWidth, threshold, imgHeight, imgWidth, s, o, dilatedImgs, maskImgs, depthImg, erodeMasks=True, beSlow=False):
    """
Compute contact masks of two objects. Each object is represented by a pair of masks, a normal and a dilated mask.
The masks are images, all of the same size.
    
Contact is taken to occur where both of the following conditions happen:
    1) there is some overlap between the dilated mask of an object A and the normal mask of the other object B AND
    2) some point on object B that is in that overlap is close to a point on object A.
        
Depth information, represented as a depth image, is used when computing closeness. The depth image has the same
size as the masks.
        
Therefore, a point on an object corresponds to a pixel on its normal mask and a pixel in the depth image. When
searching for points close to a point P, the search will look at pixels in the depth map close to the pixel 
corresponding to P.

Note that when computing closeness, only depth values are compared, rather than a full computation of distance 
between inverse projected pixels.
    
Arguments:
    searchWidth: int, controls the size of the window around a pixel to search for close pixels. The size of the
                 window is then (2*searchWidth+1)^2
    threshold: float, a threshold used to decide whether two points are close or not
    imgHeight, imgWidth: int, gives the masks and depth image dimensions
    s, o: str, name of the objects to look for contacts
    dilatedImgs, maskImgs: dict, where keys are strings and values are numpy arrays of type numpy.uint8. These
                 dictionaries should contain keys equal to string s and o
    depthImg: numpy array of type numpy.float32
    erodeMasks: bool, if True then the normal masks will be slightly eroded before searching for contacts.
    beSlow: bool. Should be kept False in most circumstances. If True, contact masks will be computed using
                 a slower implementation in python. This is/was only useful to debug the C++ implementation
                 by comparing its output to a reference.
    
Returns:
    contPS, contPO: numpy array of type numpy.uint8, contact masks for objects s and o respectively. Will have
                 the same size as the mask images. A non-zero value indicates a pixel was flagged as belonging
                 to the contact region.
    """
    def _pixelCount(mask):
        return len(mask[mask>0])
    def _expand(imgHeight, imgWidth, mask, contact):
        pixelCount = 0.2*_pixelCount(mask)
        contactCount = _pixelCount(contact)
        if 0 == contactCount:
            return contact
        while contactCount < pixelCount:
            contact = numpy.bitwise_and(mask, cv.dilate(contact, dilationKernel))
            contactCount = _pixelCount(contact)
        return contact
    contPS = numpy.zeros((imgHeight, imgWidth),dtype=numpy.uint8)
    contPO = numpy.zeros((imgHeight, imgWidth),dtype=numpy.uint8)
    maskImgS = maskImgs[s]
    maskImgO = maskImgs[o]
    if erodeMasks:
        maskImgS = cv.erode(maskImgs[s], dilationKernel)
        maskImgO = cv.erode(maskImgs[o], dilationKernel)
    if not beSlow:
        _contact(searchWidth, threshold, imgHeight, imgWidth, dilatedImgs[s], dilatedImgs[o], maskImgs[s], maskImgs[o], depthImg, contPS, contPO)
    else:
        contPS, contPO = _contactSlow(searchWidth, threshold, imgHeight, imgWidth, (dilatedImgs[s]), (dilatedImgs[o]), maskImgS, maskImgO, (depthImg), (contPS), (contPO))
    contPS = _expand(imgHeight, imgWidth, maskImgs[s], contPS)
    contPO = _expand(imgHeight, imgWidth, maskImgs[o], contPO)
    return contPS, contPO

class ContactDetection(ReifiedProcess):
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
        self._maskResults = {}
        self._depthResults = {}
        self._currentGoals = []
        self._rateMask = None
        self._droppedMask = 0
        self._rateDepth = None
        self._droppedDepth = 0
    def getGoalQueue(self):
        return self._goals
    def _checkSubscriptionRequest(self, name, queue, consumerSHM):
        return name in {"MaskImg", "DepthImg"}
    def _checkPublisherRequest(self, name, queues, consumerSHM):
        return name in {"OutImg", "DbgImg"}
    def doWork(self):
        """
        """
        def _orderQuery(q):
            if q[0]>q[1]:
                return (q[1], q[0])
            return (q[0], q[1])
        def _getMaskImgs(maskImg, results):
            results = results.get("segments", [])
            retq = {}
            for e in results:
                retq[e["type"]] = numpy.zeros(maskImg.shape,dtype=numpy.uint8)
                retq[e["type"]][maskImg == e["id"]] = 255
            return retq
        def _s2c(s):
            h = hash(s)
            b,g,r = ((h&0xFF)), ((h&0xFF00)>>8), ((h&0xFF0000)>>16)
            return (b/255.0, g/255.0, r/255.0)
        haveNew = False
        threshold = 0.05
        searchWidth = 7
        if not self._goals.empty():
            self._currentGoals, _, _ = self._goals.getWithRates()
        if not self._subscriptions["MaskImg"].empty():
            haveNew = True
            self._maskResults, self._rateMask, self._droppedMask = self._subscriptions["MaskImg"].getWithRates()
        if not self._subscriptions["DepthImg"].empty():
            haveNew = True
            self._depthResults, self._rateDepth, self._droppedDepth = self._subscriptions["DepthImg"].getWithRates()
        if haveNew and (self._maskResults.get("imgId") == self._depthResults.get("imgId")):
            with self._subscriptions["MaskImg"] as maskImg:
                with self._subscriptions["DepthImg"] as depthImg:
                    depthImgLocal = numpy.copy(depthImg)
                    maskImgs = _getMaskImgs(maskImg, self._maskResults)
            imgHeight, imgWidth = depthImgLocal.shape
            queries = set()
            searchWidth = 7
            threshold = 0.05
            dilationKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*5 + 1, 2*5 + 1), (5, 5))
            outputImg = numpy.zeros((imgHeight, imgWidth), dtype=numpy.uint32)
            results = {"imgId": self._maskResults.get("imgId"), "idx2Contact": {}, "contact2Idx": {}}
            for p,s,o in self._currentGoals:
                if "contactDistanceThreshold" == p:
                    threshold = float(s)
                elif "contactSearchWidth" == p:
                    searchWidth = int(s)
                elif "contactDilationKernel" == p:
                    dr = int(s)
                    dilationKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*dr + 1, 2*dr + 1), (dr, dr))
                elif "contactQuery" == p:
                    if s==o:
                        continue
                    if o is not None:
                        queries.add(_orderQuery((s,o)))
                    else:
                        _=[queries.add(_orderQuery((s,x))) for x in maskImgs.keys() if x!=s]
            qobjs = set([x[0] for x in queries]).union([x[1] for x in queries])
            dilatedImgs = {k: cv.dilate(maskImgs[k], dilationKernel) for k in qobjs if k in maskImgs}
            for k, (s, o) in enumerate(queries):
                if (s not in maskImgs) or (o not in maskImgs):
                    continue
                k = (k+1)*2
                idSO = k
                idOS = k-1
                results["idx2Contact"][idSO] = (s,o) 
                results["idx2Contact"][idOS] = (o,s)
                results["contact2Idx"][(s,o)] = idSO 
                results["contact2Idx"][(o,s)] = idOS 
                contPS, contPO = contact(searchWidth, threshold, imgHeight, imgWidth, s, o, dilatedImgs, maskImgs, depthImgLocal)
                outputImg[contPS>0] = idSO
                outputImg[contPO>0] = idOS
            if "OutImg" in self._publishers:
                self._publishers["OutImg"].publish(outputImg, results)
            # Do we need to prepare a debug image?
            if "DbgImg" in self._publishers:
                # Here we can hog the shared memory as long as we like -- dbgvis won't use it until we notify it that there's a new frame to show.
                with self._publishers["DbgImg"] as dbgImg:
                    workImg = dbgImg
                    if (outputImg.shape[0] != dbgImg.shape[0]) or (outputImg.shape[1] != dbgImg.shape[1]):
                        workImg = numpy.zeros((outputImg.shape[0], outputImg.shape[1], 3),dtype=dbgImg.dtype)
                    else:
                        workImg.fill(0)
                    todos = [(k, _s2c(str(k))) for k in results["idx2Contact"].keys()]
                    for k, color in todos:
                        workImg[outputImg==k]=color
                    if (outputImg.shape[0] != dbgImg.shape[0]) or (outputImg.shape[1] != dbgImg.shape[1]):
                        numpy.copyto(dbgImg, cv.resize(workImg, (dbgImg.shape[1], dbgImg.shape[0]), interpolation=cv.INTER_LINEAR))
                self._publishers["DbgImg"].sendNotifications("%.02f %.02f ifps | %d%% %d%% obj drop" % (self._rateMask if self._rateMask is not None else 0.0, self._rateDepth if self._rateDepth is not None else 0.0, self._droppedMask, self._droppedDepth))
    def cleanup(self):
        pass