import cv2 as cv
from khafre.bricks import ReifiedProcess
import numpy
import torch
from ultralytics import YOLO


class NNImgWrapper(ReifiedProcess):
    """
Subprocess in which a neural network operates on an image to produce
an image output, together with some other symbolic outputs.

Wires supported by this subprocess:
    InpImg: subscription. The input image and associated notification data.
    OutImg: publisher. The output image and associated symbolic results.
    DbgImg: publisher. An image of the segmentation masks, bboxes, and 
            detected class names.
    """
    def __init__(self):
        super().__init__()
        self._model = None
        self._dbgImg = None
    def _checkSubscriptionRequest(self, name, wire):
        return ("InpImg" == name)
    def _checkPublisherRequest(self, name, wire):
        return name in {"OutImg", "DbgImg"}
    def setModel(self, model):
        self._model = model
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
    def _customCommand(self, command):
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
            self._customCommand(command)
    def _doWork(self):
        # Do we even have a model loaded? Do we even have an output to send to?
        # Note: it is possible for the user of this process to not want an image as a result, and only a list of detections/polygons etc.
        e, inpImg, rate, dropped = self._requestSubscribedData("InpImg")
        if (self._model is not None):
            results, outputImg = self._useModel(inpImg)
            results["imgId"] = e["imgId"]
            self._requestToPublish("OutImg", results, outputImg)
            if self.havePublisher("DbgImg"):
                if self._dbgImg is None:
                    self._dbgImg = numpy.zeros(inpImg.shape, numpy.float32)
                self._prepareDbgImg(results, outputImg, self._dbgImg)
                self._requestToPublish("DbgImg", "%.02f ifps | %d%% obj drop" % (rate if rate is not None else 0.0, dropped), self._dbgImg)
    def _cleanup(self):
        self._unloadModel()
