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
    def _prepareDbgImg(self, results, outputImg, dbgImg):
        def _s2c(s):
            h = hash(s)
            b,g,r = ((h&0xFF)), ((h&0xFF00)>>8), ((h&0xFF0000)>>16)
            return (b/255.0, g/255.0, r/255.0)
        workImg = dbgImg
        if (outputImg.shape[0] != dbgImg.shape[0]) or (outputImg.shape[1] != dbgImg.shape[1]):
           workImg = numpy.zeros((outputImg.shape[0], outputImg.shape[1], 3),dtype=dbgImg.dtype)
        else:
            workImg.fill(0)
        for e in results.get("segments",[]):
            aux = cv.boundingRect(e["polygon"])
            left, top, right, bottom = [aux[0],aux[1], aux[0]+aux[2], aux[1]+aux[3]]
            cv.fillPoly(workImg, pts=[e["polygon"]], color=_s2c(e["type"]))
            (text_width, text_height), baseline = cv.getTextSize(e["type"], cv.FONT_HERSHEY_SIMPLEX, 0.5, cv.LINE_AA)
            cv.rectangle(workImg,(left,top),(right, bottom),(1.0,1.0,0.0),1)
            cv.putText(workImg, e["type"], (left, top+text_height), cv.FONT_HERSHEY_SIMPLEX, 0.5, (1.0,1.0,1.0), 1, cv.LINE_AA)
        if (outputImg.shape[0] != dbgImg.shape[0]) or (outputImg.shape[1] != dbgImg.shape[1]):
            numpy.copyto(dbgImg, cv.resize(workImg, (dbgImg.shape[1], dbgImg.shape[0]), interpolation=cv.INTER_LINEAR))
    def customCommand(self, command):
        op, args = command
        if "CONFIDENCE" == op:
            self._confidenceThreshold = args[0]
