from collections import defaultdict
import cv2 as cv
from khafre.polygons import findTopPolygons
from khafre.taskable import TaskableProcess
from khafre.tracker import Tracker
import math
import numpy
from scipy.optimize import linear_sum_assignment
from supervision import Detections, BoxAnnotator, LabelAnnotator, PolygonAnnotator
import time
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, ops


# 1. For each track, update optical flow on its stored features. Box around updated features is region of interest.
# 2. For each track, check region of interest + high confidence detections for possible matches. Collect match cost from matched flow.
# 3. For each set of tracks of same type, find optimal match.
# 4. Unmatched tracks get dormancy counter updated. If too old and not sustained, remove.
# 5. Unmatched high confidence detections initialize tracks.

class LookHarderSegmentationPredictor(SegmentationPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model="yolov8n-seg.pt", source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def postprocess(self, preds, img, orig_imgs):
    """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        for i, (pred, orig_img, img_path) in enumerate(zip(p, orig_imgs, self.batch[0])):
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results
        
class FlowTrack:
    def __init__(self, detection, imgPyramid, imgGrayscale, className, idx, settings):
        self.className = className
        self.idx = idx
        self.pyramid = imgPyramid
        self._age = 0
        self.features = None
        self._getFeatures(imgGrayscale, detection["mask"], settings)
        self._updateFromDetection(detection)
    def _updateFromDetection(self, detection):
        self.mask = None
        self.box = None
        self.conf = None
        if detection is not None:
            self.mask = detection["mask"]
            self.box = detection["box"]
            self.conf = detection["conf"]
    def _getFeatures(self, imgGrayscale, mask, settings):
        self.features = cv.goodFeaturesToTrack(imgGrayscale, mask=mask, maxCorners = settings["featureCount"], qualityLevel = settings["qualityLevel"], minDistance = settings["minDistance"], blockSize = settings["blockSize"], useHarrisDetector=False, k=0.04)
    def updateTrack(self, imgPyramid, detection, settings):
        self._updateFromDetection(detection)
        if detection is None:
            self._age += 1
            if settings["maxAge"] < self._age:
                return False
        self._age = 0
        self.features = self._getFeatures(imgGrayscale, detection["mask"], settings)
        return True
    def getCosts(self, costsMatrix, rowIdx, detections, settings):
        for colIdx, detection in detections.items():
            _, status, error = cv.calcOpticalFlowPyrLK(self.pyramid, detection["pyramid"], self.features, None, winSize=settings["winSize"], maxLevel=settings["maxLevel"], criteria = settings["stoppingCriteria"])
            error[status == 0] = settings["missingFeatureError"]
            costsMatrix[rowIdx][colIdx] = sum(error)
    def updateRegionOfInterest(self, newPyramid, settings):
        newFeatures, status, error = cv.calcOpticalFlowPyrLK(self.pyramid, newPyramid, self.features, None, winSize=settings["winSize"], maxLevel=settings["maxLevel"], criteria = settings["stoppingCriteria"])
        newFeatures = newFeatures[status==1].astype(numpy.float32)
        newFeatures = newFeatures.reshape((len(newFeatures), 2))
        left, up = numpy.min(newFeatures, 0)
        right, down = numpy.max(newFeatures, 0)
        return int(left), int(up), int(right), int(down)
    
class FlowTracker(Tracker):
    def __init__(self):
        super().__init__()
        self._bboxAnnotator = BoxAnnotator()
        self._labelAnnotator = LabelAnnotator()
        self._polygonAnnotator = PolygonAnnotator()
        self._tracks = defaultdict(list)
        self._nameSuffixes = defaultdict(0)
        self._settings["stoppingCriteria"] = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        self._settings["qualityLevel"] = 0.2
        self._settings["minDistance"] = 2
        self._settings["blockSize"] = 5
        self._settings["winSize"] = (21,21)
        self._settings["maxLevel"] = 4
        self._settings["featureCount"] = 15
        self._settings["missingFeatureError"] = 200
        #self._settings["lkParams"] = dict(winSize = (15,15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self._settings["max_det"] = 20
        self._settings["conf"] = 0.1
        self._settings["maxAge"] = 60
        self._settings["track_activation_threshold"] = 0.25
        self._settings["nmm_threshold"] = 0.65
    def _checkSubscriptionRequest(self, name, wire):
        return name in {"InpImg", "DepthImg"}
    def _checkPublisherRequest(self, name, wire):
        return name in {"MaskImg", "FlowImg", "DbgImg"}
    def _loadModel(self, modelFileName):
        self._model = YOLO(modelFileName)
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
        self._tracks = defaultdict(list)
        self._nameSuffixes = defaultdict(0)
    def _resetTracker(self, *args):
        self._startTracker()
    def _tracks2Roboflow(self):
        xyxys, masks, confidences, classIds, trackerIds = [], [], [], [], []
        for tracks in self._tracks.values():
            for track in tracks:
                if track.box is not None:
                    xyxys.append(track.box)
                    masks.append(track.mask)
                    confidences.append(track.conf)
                    classIds.append(track.className)
                    trackerIds.append(track.idx)
        return Detections(xyxy=numpy.array(xyxys), mask=numpy.array(masks), confidence=numpy.array(confidences), class_id=numpy.array(classIds), tracker_id=numpy.array(trackerIds))
    def _performStep(self):
        """
        """
        if (self._model is None):
            return

        # Step 1: retrieve and prepare inputs
        # Retrieve input data from wires, get grayscale image for optical flow, cache pyramid representation for image
        notification, inpImg, rate, dropped = self._requestSubscribedData("InpImg")
        newGrayscaleImage = cv.cvtColor(inpImage, cv.COLOR_BGR2GRAY)
        newPyramid = cv.buildOpticalFlowPyramid(newGrayscaleImage, self._settings["winSize"], self._settings["maxLevel"])

        # Step 2: retrieve regions of interest for each active track
        regionsOfInterest = {detectedType: [track.updateRegionOfInterest(newPyramid, self._settings) for track in tracks] for detectedType, tracks in self._tracks.items()}
        
        # Step 3: retrieve detections; use regions of interest to locally adjust confidence thresholds
        results = self._model(inpImg, predictor=LookHarderSegmentationPredictor, regionsOfInterest=regionsOfInterest, conf=self._settings["conf"], retina_masks=True, max_det=self._settings["max_det"], verbose=False)[0]

        detectionsByType = defaultdict(list)
        for r in zip(results.boxes.cls.tolist(), results.boxes.conf.tolist(), results.boxes.xyxyn, results.masks):
            clsIdx, conf, xyxyn, mask = r
            className = results.names[round(clsIdx)]
            adjImage = numpy.copy(newGrayscaleImage)
            adjImage[mask == 0] = 0
            pyramid = cv.buildOpticalFlowPyramid(adjImage, self._settings["winSize"], self._settings["maxLevel"])
            detectionsByType[className].append({"conf": conf, "mask": mask, "box": xyxyn, "matched": False, "pyramid": pyramid})

        # Step 4: match active tracks to detections
        matchings = {}
        for detectedType, tracks in self._tracks.items():
            if detectedType in detectionsByType:
                detections = {k:d for k, d in enumerate(detectionsByType[detectedType])}
                costsMatrix = numpy.ones((len(tracks), len(detections)))
                for k, track in enumerate(tracks):
                    track.getCosts(costsMatrix, k, detections, settings)
                track2Detection = {idTrack: detections[idDetection] for idTrack, idDetection in linear_sum_assignment(costsMatrix)}
                matchings[detectedType] = track2Detection
                for detection in track2Detection.values():
                    detection["matched"] = True
        
        # Step 5: update matched tracks and dormancy states of unmatched tracks
        for detectedType, tracks in self._tracks.items():
            self._tracks[detectedType] = [track for k, track in enumerate(tracks) if track.updateTrack(newPyramid, matchings[detectedType].get(k), self._settings)]

        # Step 6: initialize new tracks from unmatched very confident detections
        for detectedType, detections in detectionsByType.items():
            for detection in detections:
                if (not detection["matched"]) and (self._settings["track_activation_threshold"] <= detection["conf"]):
                    self._nameSuffixes[detectedType] += 1
                    self._tracks[detectedType] = FlowTrack(detection, newPyramid, newGrayscaleImage, detectedType, str(self._nameSuffixes[detectedType]), self._settings)

        # Step 7: prepare output and debug output
        outputImg = numpy.zeros(inpImg.shape, dtype=numpy.uint16)
        segments = []
        triples = set()
        
        for detectedType, tracks in self._tracks.items():
            for track in tracks:
                if track.mask is None:
                    continue
                oname = f"{track.className}_{track.idx}"
                triples.add(("isA", oname, track.className))
                col = len(triples)
            
                maskImg = numpy.zeros(track.mask.shape, dtype=numpy.uint8)
                maskImg[track.mask] = 255
                polygons = findTopPolygons(maskImg)
                segments.append({"name": oname, "type": track.className, "confidence": track.conf, "box": track.box, "polygons": polygons, "id": col})
                outputImg[track.mask] = col
        
        self._requestToPublish("MaskImg", {"segments": segments, "triples": triples}, outputImg)

        if self.havePublisher("DbgImg"):
            #masks, boxes, confidences, classNames, idxs = detections.mask, detections.xyxy, detections.confidence, detections.data.get("class_name", []), detections.tracker_id
            #labels = [f"{className}_{tracker_id}" for className, tracker_id in zip(classNames, idxs)]
            #detections = supervision.Detections.from_ultralytics(YOLOLookHarder.asUltralytics(results))
            detections, labels = self._tracks2Roboflow()
            annotatedFrame = self._bboxAnnotator.annotate(scene=inpImg.copy(), detections=detections)
            annotatedFrame = self._labelAnnotator.annotate(scene=annotatedFrame, detections=detections, labels=labels)
            # TODO: polygon annotator recomputes polygons, rewrite to avoid this
            annotatedFrame = numpy.asarray(self._polygonAnnotator.annotate(scene=annotatedFrame, detections=detections))
            dbgImg = annotatedFrame.astype(numpy.float32) / 255
            self._requestToPublish("DbgImg", "", dbgImg)
    
    def _cleanup(self):
        pass