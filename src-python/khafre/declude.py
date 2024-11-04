import cv2 as cv
from pycpd import AffineRegistration
import numpy

from khafre.taskable import TaskableProcess

def pushForward(polygons, Re, te):
    if polygons is None:
        return None
    return [numpy.dot(x, Re) + te for x in polygons]

def pullBackward(polygons, ReInv, te):
    if polygons is None:
        return None
    return [numpy.dot(x-te, ReInv) for x in polygons]

def register(polyStart, polyTarget):
    reg = AffineRegistration(**{'X': polyTarget, 'Y': polyStart})
    reg.register(lambda iteration=0, error=0, X=None, Y=None : None)
    Re, te = reg.get_registration_parameters()
    return Re, te, pushForward([polyStart], Re, te)[0]

def checkOcclusion(imgFil, imgObj, polygon, occluders):
    cv.fillPoly(imgFil, pts=[polygon], color=255)
    for e in occluders:
        cv.fillPoly(imgObj, pts=[e], color=255)
    imgObj[imgFil==0]=0
    res = cv.findContours(image=canvasTrg, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    imgObj[imgObj!=0]=0
    imgFil[imgFil!=0]=0
    # Interface change in opencv 3.2:
    # old opencv: findCountours returns contours, hierarchy
    # from 3.2: findContours returns image, contours, hierarchy
    contours, hierarchy = res[-2], res[-1]
    if hierarchy is None:
        return False, None
    return True, [x.reshape((len(x), 2)) for x, y in zip(contours, hierarchy[0]) if 0>y[3]]

def polygonUnion(imgObj, polygonsA, polygonsB):
    if polygonsA is None:
        return polygonsB
    if polygonsB is None:
        return polygonsA
    for e in polygonsA:
        cv.fillPoly(imgObj, pts=[e], color=255)
    for e in polygonsB:
        cv.fillPoly(imgObj, pts=[e], color=255)
    res = cv.findContours(image=canvasTrg, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    imgObj[imgObj!=0]=0
    # Interface change in opencv 3.2:
    # old opencv: findCountours returns contours, hierarchy
    # from 3.2: findContours returns image, contours, hierarchy
    contours, hierarchy = res[-2], res[-1]
    if hierarchy is None:
        return None
    return [x.reshape((len(x), 2)) for x, y in zip(contours, hierarchy[0]) if 0>y[3]]

def declude(imgHeight, imgWidth, objPolys, occluderPolys):
    imgFil = numpy.zeros((imgHeight, imgWidth), dtype=numpy.uint8)
    imgObj = numpy.zeros((imgHeight, imgWidth), dtype=numpy.uint8)
    frames = {}
    polyStart = None
    # Step 1: estimate affine transforms from frame to frame
    for k, (polyTarget, occluders) in enumerate(zip(objPolys, occluderPolys)):
        frames[k] = {"Re": numpy.eye(2), "ReInv": numpy.eye(2), "te": numpy.array([0.0,0.0]), "occlusion": False, "occludedPolygons": None}
        if polyStart is not None:
            Re, te, adjustedPolygon = register(polyStart, polyTarget)
            frames[k]["Re"] = Re
            frames[k]["ReInv"] = numpy.linalg.pinv(Re)
            frames[k]["te"] = te
            occlusion, occludedPolygons = checkOcclusion(imgFil, imgObj, adjustedPolygon, occluders)
            frames[k]["occlusion"] = occlusion
            # Prefer to use the polygon for frame k as start for registering k+1,
            # but in case of occlusion prefer the previous start polygon, transformed by registration
            if frames[k]["occlusion"]:
                polyTarget = adjustedPolygon
            frames[k]["occludedPolygons"] = occludedPolygons
        frames[k]["polygon"] = polyTarget
        polyStart=polyTarget
    # Step 2: merge occluded polygons and propagate forward
    last = None
    for k in range(len(frames)):
        frame = frames[k]
        last = polygonUnion(imgObj, pushForward(last, frame["Re"], frame["te"]), frame["occludedPolynomials"])
        if not frame["occlusion"]:
            frame["occludedPolynomials"] = last
    # Step 3: merge and propagate backward
    last = None
    for k in range(len(frames)-1,-1,-1):
        frame = frames[k]
        last = polygonUnion(imgObj, last, frame["occludedPolynomials"])
        if not frame["occlusion"]:
            frame["occludedPolynomials"] = last
        last = pullBackward(last, frame["ReInv"], frame["te"])
    # Return decluded polygons
    return [(k, frames[k]["occludedPolygons"]) for k in range(len(frames) if not frames[k]["occlusion"]]

'''
y = x.R+t
x = (y-t).Rinv
'''

class Decluder(Taskable):
    def __init__(self):
        super().__init__()
        self._dbgImg = None
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
        inpResults, inpImg, rateInp, droppedInp = self._requestSubscribedData("InpImg")
        maskResults, maskImg, rateMask, droppedMask = self._requestSubscribedData("MaskImg")
        masks = {s["name"]: {"polygons": s["polygons"], "confidence": s["confidence"]} for s in maskResults["segments"]}
        occImg = numpy.zeros((inpImg.shape[0], inpImg.shape[1]), dtype=numpy.uint32)
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
        for name, data in self._polygonData.items():
            for e in set().difference(relevantOverlaps.get(name, set()))
            ##### Todo
        declusionMasks = []
        triples = set()
        # Update polygons
        scratchpad = numpy.zeros(maskImg.shape, dtype=numpy.uint8)
        validPixels = numpy.zeros(maskImg.shape, dtype=numpy.uint8)
        for name, stuff in masks.items():
            if name in qObjs:
                stuff["maskIst"] = numpy.zeros(maskImg.shape, dtype=numpy.uint8)
                for p in stuff["polygons"]:
                    cv.fillPoly(stuff["maskIst"], pts = [p], color = 255)
                validPixels[stuff["maskIst"] != 0] = 255
            else:
                for p in stuff["polygons"]:
                    cv.fillPoly(validPixels, pts = [p], color = 255)
        for name, stuff in masks.items():
            if (name not in self._polygonData) or (name not in sObjs):
                self._polygonData[name] = {"polygons": stuff["polygons"], "previousOverlaps": []}
            else:
                # update soll polygonData and soll mask by affineregistering to current ist polygon, cut out invalid differences from updated soll to ist
                # using the transform from the affineregistering, update previousoverlaps polygons
                # also merge current overlap to previousoverlap
                self._polygonData[name], stuff["maskSoll"] = updatePolygons(self._polygonData[name], stuff["polygons"], scratchpad, validPixels)
            self._polygonData[name]["age"] = -1
        maskId = 0
        for k, (s, p, o) in enumerate(self._queries):
            # TODO: update previousoverlaps with current overlaps
            # Test for current occlusion of s by o
            scratchpad[masks[s]["maskSoll"] == 0] = 0
            scratchpad[masks[s]["maskSoll"] != 0] = 255
            scratchpad[masks[o]["maskIst"] == 0] = 0
            cols, counts = numpy.unique(scratchpad, return_counts=True)
            overlap = {k:v for k,v in zip(cols, counts)}[255]
            if self._settings["minOverlap"] <= overlap:
                triples.add(("occludedBy", s, o))
            # Test whether a region of s previously occluded by o is now visible
            _ = [cv.fillPoly(scratchpad, pts = [p], color = 255) for p in self._polygonData[s]["previousOverlaps"][o]]
            scratchpad[masks[s]["maskIst"] == 0] = 0
            cols, counts = numpy.unique(scratchpad, return_counts=True)
            decluded = {k:v for k,v in zip(cols, counts)}[255]
            if self._settings["minOverlap"] <= decluded:
                maskId += 1
                declusionMasks.append({"hasId": maskId, "hasP": "declusionOf", "hasS": s, "hasO": o})
                occImg[scratchpad != 0] = maskId
        # Remove polygons that are too old. Renew! Renew!
        names = list(self._polygonData.keys())
        for name in names:
            data = self._polygonData[name]
            data["age"] += 1
            if data["age"] > self._settings["maxAge"]:
                self._polygonData.pop(name)
        self._requestToPublish("OutImg", {"imgId": maskResults.get("imgId"), "masks": masks, "triples": triples}, occImg)
        # Do we need to prepare a debug image?
        if self.havePublisher("DbgImg"):
            if self._dbgImg is None:
                self._dbgImg = numpy.zeros((inpImg.shape[0], inpImg.shape[1], 3), numpy.float32)
            numpy.copyto(self._dbgImg, inpImage.astype(self._dbgImg.dtype) / 255)
            for s in sObjs:
                if s in self._polygonData:
                    polygons = self._polygonData[s]["polygons"]
                    h = hash(s)
                    color = (((h&0xFF0000) >> 16) / 255.0, ((h&0xFF00) >> 8) / 255.0, ((h&0xFF)) / 255.0)
                    cv.polylines(self._dbgImg, polygons, True, color, 3)
                    for o, polygons in self._polygonData.get(s, {"previousOverlaps": {}}):
                        h = hash((s,o))
                        color = (((h&0xFF0000) >> 16) / 255.0, ((h&0xFF00) >> 8) / 255.0, ((h&0xFF)) / 255.0)
                        cv.polylines(self._dbgImg, polygons, True, color, 3)
            self._requestToPublish("DbgImg","", self._dbgImg)
