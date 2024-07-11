import cv2 as cv
from pycpd import AffineRegistration
import numpy

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
