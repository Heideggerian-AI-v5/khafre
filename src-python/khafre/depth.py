import cv2 as cv
from khafre.nnwrappers import NNImgWrapper
from PIL import Image
import numpy

from transformers import pipeline

class TransformerDepthSegmentationWrapper(NNImgWrapper):
    """
Wrapper around a transformer monocular depth estimation model. Along the usual NNImgWrapper wires,
also listens to a command queue.

Input parameters
    device: either "cuda" or, by default, "cpu"

Wire shared memories:
    InpImg: uint8 numpy array of shape (height, width, 3)
    OutImg: float32 numpy array of shape (height, width)
    DbgImg: float32 numpy array of shape (height, width)
    """
    def __init__(self, device=None):
        super().__init__()
        if device is None:
            device = "cpu"
        self._confidenceThreshold = 0.128
        self._device = device
    def _loadModel(self, modelFileName):
        try:
            self._model = pipeline("depth-estimation", model=modelFileName, device=self._device)
        except RuntimeError as e:
            if "cpu" != self._device:
                print("Encountered exception: %s\nWill attempt to fall back to running the model on cpu, but this will be very slow!" % str(e))
                self._model = pipeline("depth-estimation", model=modelFileName, device="cpu")
            else:
                raise e
    def _useModel(self, img):
        predictions = self._model(Image.fromarray(img))
        # interpolate to original size
        outputImg = cv.resize(predictions["predicted_depth"].numpy()[0], (img.shape[1], img.shape[0]), interpolation=cv.INTER_LINEAR)
        return {}, outputImg
    def _prepareDbgImg(self, results, inputImg, outputImg):
        return cv.cvtColor(outputImg / numpy.max(outputImg), cv.COLOR_GRAY2BGR)
    def _customCommand(self, command):
        pass
