import cv2 as cv
from khafre.polygons import findTopPolygons
from khafre.taskable import TaskableProcess
import numpy

from .contactcpp import _contact

dilationKernel = cv.getStructuringElement(cv.MORPH_CROSS,(3,3),(1,1))

def _contactSlow(searchWidth, threshold, imgHeight, imgWidth, dilImgS, dilImgO, maskImgS, maskImgO, depthImg, contPS, contPO):
    """
Should not call this. A naive implementation of searching for contact masks in python. Too slow for practical use,
present here to have a kind of easier to check reference and tester for the faster implementation in C++.
    """
    def _ci(searchWidth, threshold, imgHeight, imgWidth, dilImgD, maskImgD, maskImgN, contPN):
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
            aux = _pixelCount(contact)
            if aux <= (contactCount + 10):
                break
            contactCount = aux
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

class ContactDetection(TaskableProcess):
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
        self._prefix="contact"
        self._settings["searchWidth"] = 7
        self._settings["threshold"] = 0.05
        self._settings["radius"] = 5
        self._dilationKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*5 + 1, 2*5 + 1), (5, 5))
        self._symmetricPredicates.add("contact/query")
    def _adjustDilationKernel(self, dr):
        self._dilationKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*dr + 1, 2*dr + 1), (dr, dr))
    def _checkSubscriptionRequest(self, name, wire):
        return name in {"MaskImg", "DepthImg"}
    def _checkPublisherRequest(self, name, wire):
        return name in {"OutImg", "DbgImg"}
    def _performStep(self):
        """
        """
        def _getMaskImgs(maskImg, results):
            results = results.get("segments", [])
            retq = {}
            for e in results:
                retq[e["name"]] = numpy.zeros(maskImg.shape,dtype=numpy.uint8)
                for p in e["polygons"]:
                    cv.fillPoly(retq[e["name"]], pts = [p], color = 255)
            return retq
        def _s2c(s):
            h = hash(s)
            b,g,r = ((h&0xFF)), ((h&0xFF00)>>8), ((h&0xFF0000)>>16)
            return (b/255.0, g/255.0, r/255.0)
        maskResults, maskImg, rateMask, droppedMask = self._requestSubscribedData("MaskImg")
        maskImgs = _getMaskImgs(maskImg, maskResults)
        depthResults, depthImg, rateDepth, droppedDepth = self._requestSubscribedData("DepthImg")
        print("CONTACT ObjTriples", maskResults["triples"])
        imgHeight, imgWidth = depthImg.shape
        outputImg = numpy.zeros((imgHeight, imgWidth), dtype=numpy.uint32)
        results = {"imgId": maskResults.get("imgId"), "idx2Contact": {}, "contact2Idx": {}, "triples": set(), "masks": []}
        qUniverse = [x["name"] for x in maskResults.get("segments", [])]
        self._fillInGenericQueries(qUniverse)
        qobjs = set([x[1] for x in self._queries]).union([x[2] for x in self._queries])
        dilatedImgs = {k: cv.dilate(maskImgs[k], self._dilationKernel) for k in qobjs if k in maskImgs}
        for k, (p, s, o) in enumerate(self._queries):
            if (s not in maskImgs) or (o not in maskImgs) or (s == o):
                continue
            k = (k+1)*2
            idSO = k
            idOS = k-1
            results["idx2Contact"][idSO] = (s,o)
            results["idx2Contact"][idOS] = (o,s)
            results["contact2Idx"][(s,o)] = idSO
            results["contact2Idx"][(o,s)] = idOS
            contPS, contPO = contact(self._settings["searchWidth"], self._settings["threshold"], imgHeight, imgWidth, s, o, dilatedImgs, maskImgs, depthImg)
            if (contPS>0).any():
                self._triplesFilter.addTriple(("contact", s, o))
                self._triplesFilter.addTriple(("contact", o, s))
                if self._triplesFilter.hasTriple(("contact", s, o)):
                    outputImg[contPS>0] = idSO
                    outputImg[contPO>0] = idOS
                    results["masks"].append({"hasId": idSO, "hasP": "contact", "hasS": s, "hasO": o, "polygons": findTopPolygons(contPS)})
                    results["masks"].append({"hasId": idOS, "hasP": "contact", "hasS": o, "hasO": s, "polygons": findTopPolygons(contPO)})
            else:
                self._triplesFilter.addTriple(("-contact", s, o))
                self._triplesFilter.addTriple(("-contact", o, s))
            results["triples"] = self._triplesFilter.getActiveTriples()
        self._requestToPublish("OutImg", results, outputImg)
        # Do we need to prepare a debug image?
        if self.havePublisher("DbgImg"):
            if (outputImg.shape[0] != depthImg.shape[0]) or (outputImg.shape[1] != depthImg.shape[1]):
                dbgImg = cv.resize(cv.cvtColor(depthImg / numpy.max(depthImg), cv.COLOR_GRAY2BGR), (outputImg.shape[1], outputImg.shape[0]), interpolation=cv.INTER_LINEAR)
            else:
                dbgImg = numpy.copy(cv.cvtColor(depthImg / numpy.max(depthImg), cv.COLOR_GRAY2BGR)) 
            todos = [(k, _s2c(str(k))) for k in results["idx2Contact"].keys()]
            for k, color in todos:
                dbgImg[outputImg==k]=color
            self._requestToPublish("DbgImg", "%.02f %.02f ifps | %d%% %d%% obj drop" % (rateMask if rateMask is not None else 0.0, rateDepth if rateDepth is not None else 0.0, droppedMask, droppedDepth), dbgImg)
    def _cleanup(self):
        pass
