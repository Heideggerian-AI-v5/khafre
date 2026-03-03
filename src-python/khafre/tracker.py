import cv2 as cv
from khafre.bricks import ReifiedProcess
from khafre.polygons import findTopPolygons
from khafre.taskable import TaskableProcess
import numpy
import supervision
import time
import torch
from ultralytics import YOLO

from supervision.config import CLASS_NAME_DATA_FIELD

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
#from transformers import AutoModelForMaskGeneration, pipeline

from transformers import Sam2VideoModel, Sam2VideoProcessor

from accelerate import Accelerator

class SegmentationWrapper:
    def __init__(self):
        pass
    def _loadModel(self, *args):
        pass
    def _unloadModel(self):
        pass
    def detect(self, img, goals):
        pass
    def isLoaded(self, img):
        return False

class YOLOSegmentationWrapper(SegmentationWrapper):
    def __init__(self):
        super().__init__()
        self._model = None
    def _loadModel(self, modelFileName):
        self._model = YOLO(modelFileName)
    def _unloadModel(self, model):
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
    def isLoaded(self):
        return self._model is not None
    def detect(self, inpImg, goals):
        #results = self._model(inpImg, conf=self._settings["conf"], retina_masks=True, max_det=self._settings["max_det"], verbose=False)[0]
        results = self._model(inpImg, conf=0.05, retina_masks=True, max_det=80, verbose=False)[0]
        
        #names = [results.names[round(x)] for x in (results.boxes.cls.tolist())]
        #confs = [x for x in (results.boxes.conf.tolist())]
                
        return supervision.Detections.from_ultralytics(results)

class SegmentAnythingWrapper(SegmentationWrapper):
    def __init__(self):
        super().__init__()
        self._device = None
        self._detector_model = None
        self._segmenter_model = None
        self._detector_processor = None
        self._segmenter_processor = None
        self._detector_model_id = None
        self._segmenter_model_id = None
    def _loadModel(self, detector_model_id, segmenter_model_id):
        self._detector_model_id = "IDEA-Research/grounding-dino-tiny"
        if detector_model_id is not None:
            self._detector_model_id = detector_model_id
        self._segmenter_model_id = "facebook/sam2.1-hiera-tiny"
        if segmenter_model_id is not None:
            self._segmenter_model_id = segmenter_model_id
        self._device = Accelerator().device
        self._detector_processor = AutoProcessor.from_pretrained(self._detector_model_id)
        self._detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(self._detector_model_id).to(self._device)
        self._segmenter_model = Sam2VideoModel.from_pretrained(self._segmenter_model_id).to(self._device, dtype=torch.bfloat16)
        self._segmenter_processor = Sam2VideoProcessor.from_pretrained(self._segmenter_model_id)
    def _unloadModel(self, model):
        if self._detector_model is not None:
            del self._detector_model
            self._detector_model = None
            del self._segmenter_model
            self._segmenter_model = None
            del self._detector_processor
            self._detector_processor = None
            del self._segmenter_processor
            self._segmenter_processor = None
            torch.cuda.empty_cache()
    def isLoaded(self):
        return self._detector_model is not None
    def detect(self, inpImg, goals):
        text_labels = []
        labelMap = {}
        invLabelMap = {}
        for p, s, _ in goals:
            if "find" == p:
                text_labels.append(s.replace("_", " "))
                labelMap[text_labels[-1]] = len(labelMap)
                invLabelMap[len(invLabelMap)] = text_labels[-1]
        
        inputs = self._detector_processor(images=inpImg, text=text_labels, return_tensors="pt").to(self._detector_model.device)
        with torch.no_grad():
            outputs = self._detector_model(**inputs)
        
        #detection_results = self._detector_processor.post_process_grounded_object_detection(outputs,inputs.input_ids,text_threshold=0.3,target_sizes=[inpImg.size[::-1]])[0]#,threshold=0.4
        detection_results = self._detector_processor.post_process_grounded_object_detection(outputs,inputs.input_ids,text_threshold=0.3,target_sizes=[inpImg.shape[:2]])[0]#,threshold=0.4
        print(detection_results, type(detection_results))
        #boxes = []
        #for result in detection_results:
        #    xyxy = result.box.xyxy
        #    boxes.append(xyxy)
        
        #inputs = self._segmenter_processor(images=inpImg, input_boxes=[boxes], return_tensors="pt").to(self._segmenter_model.device)
        inference_session = self._segmenter_processor.init_video_session(inference_device=self._device, dtype=torch.bfloat16)
    
        #inputs = self._segmenter_processor(images=inpImg, input_boxes=[detection_results["boxes"].cpu()], return_tensors="pt").to(self._segmenter_model.device)
        inputs = self._segmenter_processor(images=inpImg, device=self._segmenter_model.device, return_tensors="pt")
        self._segmenter_processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx = 0,
            obj_ids = list(range(detection_results["boxes"].shape[0])),
            input_boxes=[detection_results["boxes"].cpu()],
            original_size=inputs.original_sizes[0], # need to be provided when using streaming video inference
        )

        with torch.no_grad():
            #outputs = self._segmenter_model(**inputs)
            outputs = self._segmenter_model(inference_session=inference_session, frame=inputs.pixel_values[0])
        
        masks = self._segmenter_processor.post_process_masks(masks=[outputs.pred_masks], original_sizes=inputs.original_sizes, reshaped_input_sizes=inputs.reshaped_input_sizes)[0]
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        #masks = masks.numpy().astype(numpy.uint8)
        #masks = list(masks)
        #for idx, mask in enumerate(masks):
        #    shape = mask.shape
        #    polygon = mask_to_polygon(mask)
        #    mask = polygon_to_mask(polygon, shape)
        #    masks[idx] = mask

        #for detection_result, mask in zip(detection_results, masks):
        #    detection_result.mask = mask
        detection_results["masks"] = masks

        detection_results["labels"] = torch.Tensor([labelMap.get(x, 0) for x in detection_results["labels"]])

        boxes = None
        if "boxes" in detection_results:
            boxes = detection_results["boxes"].cpu().detach().numpy()
        masks = detection_results["masks"].cpu().detach().numpy().astype(bool)
        class_ids = detection_results["labels"].cpu().detach().numpy().astype(int)

        data = {}

        if invLabelMap is not None:
            class_names = numpy.array([invLabelMap[class_id] for class_id in class_ids])
            data[CLASS_NAME_DATA_FIELD] = class_names

        return supervision.Detections(xyxy=boxes, mask=masks, confidence=detection_results["scores"].cpu().detach().numpy(), class_id=class_ids, data=data)
        #return supervision.Detections.from_transformers(transformers_results=detection_results, id2label=invLabelMap)

class Tracker(TaskableProcess):
    """
Subprocess which wraps some object tracking method.

Wires supported by this subprocess:
    InpImg: subscription. The input image and associated notification data, i.e. object detections.
    OutImg: publisher. The output image and associated symbolic results.
    DbgImg: publisher. An image of the segmentation masks, bboxes, and 
            detected class names.
    """
    def __init__(self, segmenter=None):
        super().__init__()
        self._prefix="tracker"
        if segmenter is None:
            segmenter = YOLOSegmentationWrapper()
        if not isinstance(segmenter, SegmentationWrapper):
            raise ValueError("A tracker's segmenter must be a SegmentationWrapper object or None")
        self._segmenter = segmenter
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
    def _customCommand(self, command):
        op, args = command
        if "START" == op:
            self._startTracker(*args)
        elif "RESET" == op:
            self._resetTracker(*args)
        elif "LOAD" == op:
            self._segmenter._loadModel(*args)
        elif "UNLOAD" == op:
            self._segmenter._unloadModel(*args)
        else:
            self._customCommandInternal(command)
    def _startTracker(self, *args):
        pass
    def _resetTracker(self, *args):
        pass

class ByteTracker(Tracker):
    def __init__(self, segmenter=None):
        super().__init__(segmenter)
        self._bboxAnnotator = supervision.BoxAnnotator()
        self._labelAnnotator = supervision.LabelAnnotator()
        self._polygonAnnotator = supervision.PolygonAnnotator()
        self._settings["max_det"] = 80 # TODO: reconnect params to segmenters, e.g. max_det and conf to YOLO
        self._settings["conf"] = 0.05
        self._settings["frame_rate"] = 30
        self._settings["lost_track_buffer"] = 60
        self._settings["max_time_lost"] = int((self._settings["frame_rate"] / 30) * self._settings["lost_track_buffer"])
        self._settings["minimum_matching_threshold"] = 0.8
        self._settings["minimum_consecutive_frames"] = 1
        self._settings["track_activation_threshold"] = 0.3
        self._settings["det_interval"] = 0.1
        self._settings["det_thresh"] = (self._settings["track_activation_threshold"] + self._settings["det_interval"])
        self._settings["nmm_threshold"] = 0.35
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
        if (self._tracker is None) or (not self._segmenter.isLoaded()):
            return
        
        notification, inpImg, rate, dropped = self._requestSubscribedData("InpImg")
        print("Start tracker on", notification.get("imgId"))
        
        #print(self._currentGoals)
        detections = self._segmenter.detect(inpImg, self._queries)
        
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
                className = className.replace(" ", "_")
                oname = f"{className}_{idx}"
                triples.add(("isA", oname, className))
                col = len(triples)
            
                maskImg = numpy.zeros(mask.shape, dtype=numpy.uint8)
                maskImg[mask] = 255
                polygons = findTopPolygons(maskImg)
                segments.append({"name": oname, "type": className, "confidence": confidence, "box": box, "polygons": polygons, "id": col})
                
                for p in polygons:
                    cv.fillPoly(outputImg, pts = [p], color = col)

        self._requestToPublish("OutImg", {"segments": segments, "triples": triples, "imgId": notification.get("imgId")}, outputImg)

        if self.havePublisher("DbgImg"):
            annotatedFrame = self._bboxAnnotator.annotate(scene=inpImg.copy(), detections=detections)
            annotatedFrame = self._labelAnnotator.annotate(scene=annotatedFrame, detections=detections, labels=labels)
            annotatedFrame = numpy.asarray(self._polygonAnnotator.annotate(scene=annotatedFrame, detections=detections))
            dbgImg = annotatedFrame.astype(numpy.float32) / 255
            self._requestToPublish("DbgImg", "", dbgImg)
        print("Ended tracker on", notification.get("imgId"))

