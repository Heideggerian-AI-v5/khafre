import cv2 as cv

def findPolygons(img):
    contours, hierarchy = cv.findContours(image=img, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def findTopPolygons(img):
    contours, hierarchy = findPolygons(img)
    if hierarchy is None:
        return []
    contours = [x.reshape((len(x), 2)) for x in contours]
    polygons = []
    for polygon, h in zip(contours, hierarchy[0]):
        if (0 > h[3]) and (2 < len(polygon)):
            polygons.append(polygon)
    return polygons

