import cv2 as cv
from khafre.nnwrappers import NNImgWrapper
import numpy
from ultralytics import YOLO

class YOLOObjectSegmentationWrapper(NNImgWrapper):
    """
Wrapper around a YOLO object segmentation model. Along the usual NNImgWrapper wires,
also listens to a command queue.

Wire shared memories:
    InpImg: uint8 numpy array of shape (height, width, 3)
    OutImg: uint16 numpy array of shape (height, width)
    DbgImg: float32 numpy array of shape (height, width, 3)
    """
    def __init__(self):
        super().__init__()
        self._confidenceThreshold = 0.128
    def _loadModel(self, modelFileName):
        self._model = YOLO(modelFileName)
    def _useModel(self, img):
        def _clip(x, height, width):
            x = numpy.array(x, dtype=numpy.int32)
            x[:,0] = numpy.clip(x[:,0],0,width-1)
            x[:,1] = numpy.clip(x[:,1],0,height-1)
            return x
        height = img.shape[0]
        width = img.shape[1]
        outputImg = numpy.zeros((height, width), dtype=numpy.uint16)
        results = self._model(img, conf=self._confidenceThreshold, verbose=False)[0] # For some reason YOLO returns a list but it seems only the first element matters.
        if results.masks is None:
            return {}, outputImg
        polys = [_clip(x, height, width) for x in results.masks.xy]
        names = [results.names[round(x)] for x, p in zip(results.boxes.cls.tolist(), polys) if (2<len(p))]
        confs = [x for x,p in zip(results.boxes.conf.tolist(), polys) if (2<len(p))]
        boxes = [x for x,p in zip(results.boxes.xyxyn.tolist(), polys) if (2<len(p))]
        polys = [p for p in polys if (2<len(p))] 
        retq = [{"type": t, "confidence": c, "box": b, "polygon": p, "id": k+1} for k,(t,c,b,p) in enumerate(zip(names, confs, boxes, polys))]
        for k, p in enumerate(polys):
            cv.fillPoly(outputImg, pts = [p], color = k+1)
        return {"segments": retq}, outputImg
    def _sortByArea(self, segments):
        def _area(segment):
            return -(segment["box"][2]-segment["box"][0])*(segment["box"][3]-segment["box"][1])            
        return [y[1] for y in sorted([(_area(x), x) for x in segments], key=lambda x: x[0])]
    def _prepareDbgImg(self, results, inputImg, outputImg):
        def _scaleSegment(segment, factors):
            retq["type"] = segment["type"]
            retq["box"] = segment["box"]
            retq["polygon"] = numpy.column_stack([segment["polygon"][:,0]*factors[0], segment["polygon"][:,1]*factors[1]])
            return retq
        def _safeDiv(a,b):
            if 0 == b:
                return 0
            return a/b
        def _s2c(s):
            h = hash(s)
            b,g,r = ((h&0xFF)), ((h&0xFF00)>>8), ((h&0xFF0000)>>16)
            return (b/255.0, g/255.0, r/255.0)
        dbgImg = inputImg.astype(numpy.float32) / 255
        workSegments = results.get("segments", [])
        sortedSegments = self._sortByArea(workSegments)
        for e in sortedSegments:
            left, top, right, bottom = int(e["box"][0]*dbgImg.shape[1]), int(e["box"][1]*dbgImg.shape[0]), int(e["box"][2]*dbgImg.shape[1]), int(e["box"][3]*dbgImg.shape[0])
            cv.polylines(dbgImg, [e["polygon"]], True, _s2c(e["type"]), 3)
            (text_width, text_height), baseline = cv.getTextSize(e["type"], cv.FONT_HERSHEY_SIMPLEX, 0.5, cv.LINE_AA)
            cv.rectangle(dbgImg,(left,top),(right, bottom),(1.0,1.0,0.0),1)
            cv.putText(dbgImg, e["type"], (left, top+text_height), cv.FONT_HERSHEY_SIMPLEX, 0.5, (1.0,1.0,1.0), 1, cv.LINE_AA)
        return dbgImg
    def _customCommand(self, command):
        op, args = command
        if "CONFIDENCE" == op:
            self._confidenceThreshold = args[0]
