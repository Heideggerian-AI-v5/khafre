import cv2 as cv
from khafre.bricks import ReifiedProcess, RatedSimpleQueue, NameTaken
from multiprocessing import Lock, shared_memory, Queue
import numpy
import torch
from ultralytics import YOLO


class NNImgWrapper(ReifiedProcess):
    """
    """
    def __init__(self):
        super().__init__()
        self._inputImage = None
        self._inputNotification = None
        self._model = None
        self._command = Queue()
        self._outputImage = None
        self._outputNotification = None
        self._dbgImage = None
        self._dbgNotification = None
    def setInputImagePort(self, consumerSHMPort, inputNotificationQueue):
        self._inputImage = consumerSHMPort
        self._inputNotification = inputNotificationQueue
    def setModel(self, model):
        self._model = model
    def sendCommand(self, command, block=False, timeout=None):
        self._command.put(command, block=block, timeout=timeout)
    def setOutputImagePort(self, consumerSHMPort, outputNotificationQueue):
        self._outputImage = consumerSHMPort
        self._outputNotification = outputNotificationQueue
    def setOutputDbgImagePort(self, consumerSHMPort, dbgNotificationQueue):
        self._dbgImage = consumerSHMPort
        self._dbgNotification = dbgNotificationQueue
    def _unloadModel(self):
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
    def _loadModel(self, modelFileName):
        """
Subclasses should overload this with code appropriate to load a neural model.
        """
        pass
    def _useModel(self, img):
        """
Subclasses should overload this with code appropriate to use a neural model.
        """
        pass
    def _prepareDbgImg(self, results, img, dbgImg):
        """
Subclasses should overload this with code appropriate to preparing a dbg image
from the neural network results.        
        """
        pass
    def customCommand(self, command):
        """
Subclasses should implement command handling code here.

(Un)Load model commands are handled by the base class code.
        """
        pass
    def _handleCommand(self, command):
        op, args = command
        if "LOAD" == op:
            self._loadModel(*args)
        elif "UNLOAD" == op:
            self._unloadModel()
        else:
            self.customCommand(command)
    def doWork(self):
        """
        """
        # First, check for commands.
        # This may result in (re)loading a model, and will take time.
        # Expect some frame drops when that happens.
        while not self._command.empty():
            self._handleCommand(self._command.get())
        # Do we even have a model loaded? Do we even have an output to send to?
        # Note: it is possible for the user of this process to not want an image as a result, and only a list of detections/polygons etc.
        if (self._model is not None) and (self._outputNotification is not None):
            # Do we even have an image to work on?
            if not self._inputNotification.empty():
                # We only get the latest image -- need to check how many were dropped along the way.
                _,rate,dropped = self._inputNotification.getWithRates()
                # Get a copy of the image so we can free it for others (e.g., the image acquisition process) as soon as possible.
                with self._inputImage as inpImg:
                    ourImg = numpy.copy(inpImg)
                results, outputImg = self._useModel(ourImg)
                if self._outputImage is not None:
                    self._outputImgage.send(outputImg)
                self._outputNotification.put(results)
                # Do we need to prepare a debug image?
                if self._dbgImage is not None:
                    # Here we can hog the shared memory as long as we like -- dbgvis won't use it until we notify it that there's a new frame to show.
                    with self._dbgImage as dbgImg:
                        self._prepareDbgImg(results, outputImg, dbgImg)
                    self._dbgNotification.put("%.02f ifps | %d%% obj drop" % (rate if rate is not None else 0.0, dropped))
    def cleanup(self):
        self._unloadModel()
        while not self._inputNotification.empty():
            self._inputNotification.get()

class YOLOObjectSegmentationWrapper(NNImgWrapper):
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
            return [], outputImg
        polys = [_clip(x, height, width) for x in results.masks.xy]
        names = [results.names[round(x)] for x, p in zip(results.boxes.cls.tolist(), polys) if (2<len(p))]
        confs = [x for x,p in zip(results.boxes.conf.tolist(), polys) if (2<len(p))]
        boxes = [x for x,p in zip(results.boxes.xyxyn.tolist(), polys) if (2<len(p))]
        polys = [p for p in polys if (2<len(p))] 
        retq = [{"type": t, "confidence": c, "box": b, "polygon": p, "id": k+1} for k,(t,c,b,p) in enumerate(zip(names, confs, boxes, polys))]
        for k, p in polys:
            cv.fillPoly(outputImg, pts = [p], color = k+1)
        return retq, outputImg
    def _prepareDbgImg(self, results, outputImg, dbgImg):
        def _s2c(s):
            h = hash(s)
            b,g,r = ((h&0xFF)), ((h&0xFF00)>>8), ((h&0xFF0000)>>16)
            return (b/255.0, g/255.0, r/255.0)
        workImg = dbgImg
        if (outputImg.shape[0] != dbgImg.shape[0]) or (outputImg.shape[1] != dbgImg.shape[1]):
           workImg = numpy.zeros((outputImg.shape[0], outputImg.shape[1], 3),dtype=dbgImg.dtype)
        for e in results:
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
    
