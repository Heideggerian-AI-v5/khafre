import cv2 as cv
from khafre.bricks import ReifiedProcess
from multiprocessing import Queue
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
        #self._inputImage = None
        #self._inputNotification = None
        self._model = None
        self._command = Queue()
        #self._outputImage = None
        #self._outputNotification = None
        #self._dbgImage = None
        #self._dbgNotification = None
    def _checkSubscriptionRequest(self, name, queue, consumerSHM):
        return ("InpImg" == name)
    def _checkPublisherRequest(self, name, queues, consumerSHM):
        return name in {"OutImg", "DbgImg"}
    def setModel(self, model):
        self._model = model
    def sendCommand(self, command, block=False, timeout=None):
        self._command.put(command, block=block, timeout=timeout)
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
        if (self._model is not None):
            # Do we even have an image to work on?
            if not self._subscriptions["InpImg"].empty():
                # We only get the latest image -- need to check how many were dropped along the way.
                _,rate,dropped = self._subscriptions["InpImg"].getWithRates()
                # Get a copy of the image so we can free it for others (e.g., the image acquisition process) as soon as possible.
                with self._subscriptions["InpImg"] as inpImg:
                    ourImg = numpy.copy(inpImg)
                results, outputImg = self._useModel(ourImg)
                if "OutImg" in self._publishers:
                    self._publishers["OutImg"].publish(outputImg, results)
                # Do we need to prepare a debug image?
                if "DbgImg" in self._publishers:
                    # Here we can hog the shared memory as long as we like -- dbgvis won't use it until we notify it that there's a new frame to show.
                    with self._publishers["DbgImg"] as dbgImg:
                        self._prepareDbgImg(results, outputImg, dbgImg)
                    self._publishers["DbgImg"].sendNotifications("%.02f ifps | %d%% obj drop" % (rate if rate is not None else 0.0, dropped))
    def cleanup(self):
        self._unloadModel()
        while not self._inputNotification.empty():
            self._inputNotification.get()
