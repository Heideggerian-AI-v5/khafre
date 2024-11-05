import cv2 as cv
from khafre.bricks import ReifiedProcess
from khafre.polygons import findTopPolygons
from khafre.taskable import TaskableProcess
import numpy
import supervision
import time
import torch
from ultralytics import YOLO

class Tracker(TaskableProcess):
    """
Subprocess which wraps some object tracking method.

Wires supported by this subprocess:
    InpImg: subscription. The input image and associated notification data, i.e. object detections.
    OutImg: publisher. The output image and associated symbolic results.
    DbgImg: publisher. An image of the segmentation masks, bboxes, and 
            detected class names.
    """
    def __init__(self):
        super().__init__()
        self._prefix="tracker"
        self._model = None
    def _checkSubscriptionRequest(self, name, wire):
        return ("InpImg" == name)
    def _checkPublisherRequest(self, name, wire):
        return name in {"OutImg", "DbgImg"}
    def _customCommandInternal(self, command):
        """
Subclasses should implement command handling code here.

(Un)Load model commands are handled by the base class code.
        """
        pass
    def _loadModel(self, model):
        pass
    def _unloadModel(self, model):
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
    def _customCommand(self, command):
        op, args = command
        if "START" == op:
            self._startTracker(*args)
        elif "RESET" == op:
            self._resetTracker(*args)
        elif "LOAD" == op:
            self._loadModel(*args)
        elif "UNLOAD" == op:
            self._unloadModel(*args)
        else:
            self._customCommandInternal(command)
    def _startTracker(self, *args):
        pass
    def _resetTracker(self, *args):
        pass

class ByteTracker(Tracker):
    def __init__(self):
        super().__init__()
        self._bboxAnnotator = supervision.BoxAnnotator()
        self._labelAnnotator = supervision.LabelAnnotator()
        self._polygonAnnotator = supervision.PolygonAnnotator()
        self._settings["conf"] = 0.25
        self._settings["frame_rate"] = 30
        self._settings["lost_track_buffer"] = 30
        self._settings["max_time_lost"] = int((self._settings["frame_rate"] / 30) * self._settings["lost_track_buffer"])
        self._settings["minimum_matching_threshold"] = 0.8
        self._settings["minimum_consecutive_frames"] = 1
        self._settings["track_activation_threshold"] = 0.25
        self._settings["det_interval"] = 0.1
        self._settings["det_thresh"] = (self._settings["track_activation_threshold"] + self._settings["det_interval"])
        self._settings["nmm_threshold"] = 0.65
        self._tracker = None
        self._onSettingsUpdate["frame_rate"] = self._updateFrameRate
        self._onSettingsUpdate["lost_track_buffer"] = self._updateLostTrackBuffer
        self._onSettingsUpdate["max_time_lost"] = self._updateMaxTimeLost
        self._onSettingsUpdate["minimum_matching_threshold"] = self._updateMinimumMatchingThreshold
        self._onSettingsUpdate["minimum_consecutive_frames"] = self._updateMinimumConsecutiveFrames
        self._onSettingsUpdate["track_activation_threshold"] = self._updateTrackActivationThreshold
        self._onSettingsUpdate["det_interval"] = self._updateDetInterval
        self._onSettingsUpdate["det_thresh"] = self._updateDetThresh
    def _updateFrameRate(self, x):
        self._settings["frame_rate"] = x
        if self._tracker is not None:
            self._tracker.max_time_lost = int((self._settings["frame_rate"] / 30) * self._settings["lost_track_buffer"])
    def _updateLostTrackBuffer(self, x):
        self._settings["lost_track_buffer"] = x
        if self._tracker is not None:
            self._tracker.max_time_lost = int((self._settings["frame_rate"] / 30) * self._settings["lost_track_buffer"])
    def _updateMaxTimeLost(self, x):
        self._settings["max_time_lost"] = x
        if self._tracker is not None:
            self._tracker.max_time_lost = x
    def _updateMinimumMatchingThreshold(self, x):
        self._settings["minimum_matching_threshold"] = x
        if self._tracker is not None:
            self._tracker.minimum_matching_threshold = x
    def _updateMinimumConsecutiveFrames(self, x):
        self._settings["minimum_consecutive_frames"] = x
        if self._tracker is not None:
            self._tracker.minimum_consecutive_frames = x
    def _updateTrackActivationThreshold(self, x):
        self._settings["track_activation_threshold"] = x
        if self._tracker is not None:
            self._tracker.track_activation_threshold = x
            self._tracker.det_thresh = (self._settings["track_activation_threshold"] + self._settings["det_interval"])
    def _updateDetInterval(self, x):
        self._settings["det_interval"] = x
        if self._tracker is not None:
            self._tracker.det_thresh = (self._settings["track_activation_threshold"] + self._settings["det_interval"])
    def _updateDetThresh(self, x):
        self._settings["det_thresh"] = x
        if self._tracker is not None:
            self._tracker.det_thresh = x
            self._tracker.track_activation_threshold = (self._settings["det_thresh"] - self._settings["det_interval"])
    def _loadModel(self, modelFileName):
        self._model = YOLO(modelFileName)
    def _startTracker(self, *args):
        if 0 < len(args):
            d = args[0]
            if isinstance(d, dict):
                for k in ["frame_rate", "lost_track_buffer", "max_time_lost", "minimum_matching_threshold", "minimum_consecutive_frames", "track_activation_threshold", "det_interval", "det_thresh"]:
                    if k in d:
                        self._settings[k] = d[k]
        self._tracker = supervision.ByteTrack(track_activation_threshold=self._settings["track_activation_threshold"],
                                              lost_track_buffer=self._settings["lost_track_buffer"],
                                              minimum_matching_threshold=self._settings["minimum_matching_threshold"],
                                              frame_rate=self._settings["frame_rate"],
                                              minimum_consecutive_frames=self._settings["minimum_consecutive_frames"])
    def _resetTracker(self, *args):
        if self._tracker is not None:
            self._tracker.reset()
        else:
            self._startTracker(*args)
    def _performStep(self):
        if (self._tracker) is None or (self._model is None):
            return
        
        notification, inpImg, rate, dropped = self._requestSubscribedData("InpImg")
        results = self._model(inpImg, conf=self._settings["conf"], verbose=False)[0]
                
        detections = supervision.Detections.from_ultralytics(results)
        detections = detections.with_nmm(threshold=self._settings["nmm_threshold"])
        detections = self._tracker.update_with_detections(detections)
        
        masks, boxes, confidences, classNames, idxs = detections.mask, detections.xyxy, detections.confidence, detections.data.get("class_name", []), detections.tracker_id
        labels = [f"{className}_{tracker_id}" for className, tracker_id in zip(classNames, idxs)]
        
        height = inpImg.shape[0]
        width = inpImg.shape[1]
        outputImg = numpy.zeros((height, width), dtype=numpy.uint16)
        
        segments = []
        triples = set()
        if masks is not None:
            for mask, box, confidence, className, idx in zip(masks, boxes, confidences, classNames, idxs):
                if mask is None:
                    continue
                oname = f"{className}_{idx}"
                triples.add(("isA", oname, className))
                col = len(triples)
            
                maskImg = numpy.zeros(mask.shape, dtype=numpy.uint8)
                maskImg[mask] = 255
                polygons = findTopPolygons(maskImg)
                segments.append({"name": oname, "type": className, "confidence": confidence, "box": box, "polygons": polygons, "id": col})
                
                for p in polygons:
                    cv.fillPoly(outputImg, pts = [p], color = col)

        self._requestToPublish("OutImg", {"segments": segments, "triples": triples}, outputImg)

        if self.havePublisher("DbgImg"):
            annotatedFrame = self._bboxAnnotator.annotate(scene=inpImg.copy(), detections=detections)
            annotatedFrame = self._labelAnnotator.annotate(scene=annotatedFrame, detections=detections, labels=labels)
            annotatedFrame = numpy.asarray(self._polygonAnnotator.annotate(scene=annotatedFrame, detections=detections))
            dbgImg = annotatedFrame.astype(numpy.float32) / 255
            self._requestToPublish("DbgImg", "", dbgImg)

