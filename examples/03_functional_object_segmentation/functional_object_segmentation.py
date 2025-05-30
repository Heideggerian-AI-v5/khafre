import argparse
import cv2 as cv
import numpy
import os
import sys
import time

from khafre.bricks import drawWire, setSignalHandlers, startKhafreProcesses, stopKhafreProcesses
from khafre.dbgvis import DbgVisualizer
from khafre.declude import Decluder
from khafre.depth import TransformerDepthSegmentationWrapper
from khafre.tracker import ByteTracker
from khafre.contact import ContactDetection
from khafre.optical_flow import OpticalFlow
from khafre.reasoning import Reasoner
from khafre.framestore import YOLOFrameSaver
from khafre.videocapture import RecordedVideoFeed

### Functional object segmentation example:
# based on observed behavior, such as contacts, occlusions, relative movements,
# selects regions of objects as being relevant for some particular task.
# In this example, we look at areas of cutlery that contact and are occluded by fruit,
# which is interpreted as those areas being relevant for cutting and piercing tasks.


def main():

    ######### Part 0: some general purpose setup. For khafre usage examples, scroll down to Part 1.
    
    # A neat trick of general applicability: sometimes we may wish to access resources that are predictably placed relative to the
    # running python script (for example, the theory files below). Just using relative paths in this case is not a robust solution:
    # relative paths are resolved from the current working directory, i.e. the active directory in the command line in which python runs.
    # This means running the script from another folder would invalidate the relative paths to resources.
    # The following line however allows us to get the path to the folder containing the running script, and from there we have reliable paths to
    # our resources.
    basePath = os.path.dirname(os.path.abspath(__file__)).replace("\\","/")

    # Setting up a command line argument parser and reading the provided arguments.
    parser = argparse.ArgumentParser(prog='functional_object_segmentation', description='Perform functional object segmentation to detect affordance-related parts of objects.', epilog='')
    parser.add_argument('-iv', '--input_video', default="", help="REQUIRED: Path to the video file that will be used as input for khafre.")
    parser.add_argument('-nogui', '--nogui', action='store_true', help='OPTIONAL: Flag to disable debug visualization. Will also disable early termination via ESCAPE keypress (early termination via keyboard interrupt or terminate signal remains available). Use this on systems with no window manager. You will likely need this when running khafre on remote machines too.')
    parser.add_argument('-efp', '--event_frames_path', default="./event_frames", help="OPTIONAL: Path to the folder where khafre will store selected frames and their image schematic descriptions.")
    parser.add_argument('-sfp', '--stored_frames_path', default="./stored_training_frames", help="OPTIONAL: Path to the folder where khafre will store selected frames for training functional object part recognition and their annotations.")
    parser.add_argument('-ml', '--max_lines', default="640", help="OPTIONAL: Maximum number of lines for a video frame. If the input video has higher resolution, it will be downscaled so that its frame height matches this parameter. Default is 640. If parameter given, must be convertible to integer.")
    
    arguments = parser.parse_args()
    
    headless = arguments.nogui
    if not headless:
        from khafre.utils import repeatUntilKey
    
    try:
        maxLines = int(arguments.max_lines)
    except Exception as e:
        print("Encountered exception while reading the max lines parameter.")
        print(e)
        print("Will now exit.")
        sys.exit(0)
        
    try:
        inputVideo = os.path.abspath(arguments.input_video).replace("\\","/")
        if ("" == inputVideo) or (not os.path.isfile(inputVideo)):
            raise ValueError
        vid = cv.VideoCapture(inputVideo)
        imgWidth, imgHeight = int(vid.get(cv.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
        if maxLines < imgHeight:
            imgWidth, imgHeight = int(imgWidth*maxLines/imgHeight), maxLines
    except Exception as e:
        print("Must provide a valid input file!")
        print(e)
        print("Will now exit.")
        sys.exit(0)
    
    eventPath = os.path.abspath(arguments.event_frames_path).replace("\\","/")
    
    if not os.path.isdir(eventPath):
        try:
            os.mkdir(eventPath)
        except Exception as e:
            print("Encountered exception during an attempt to create a folder for the selected frames.")
            print(e)
            print("Will now exit.")
            sys.exit(0)

    storagePath = os.path.abspath(arguments.stored_frames_path).replace("\\","/")
    
    if not os.path.isdir(storagePath):
        try:
            os.mkdir(storagePath)
        except Exception as e:
            print("Encountered exception during an attempt to create a folder for the training frames.")
            print(e)
            print("Will now exit.")
            sys.exit(0)

    percIntTheory = os.path.join(basePath,"theories/example_cutlery_and_fruit/perceptionInterpretation.dfl")
    updSchTheory = os.path.join(basePath,"theories/example_cutlery_and_fruit/updateSchemas.dfl")
    connQTheory = os.path.join(basePath,"theories/example_cutlery_and_fruit/connectivityQueries.dfl")
    schClsTheory = os.path.join(basePath,"theories/example_cutlery_and_fruit/schemaClosure.dfl")
    schIntTheory = os.path.join(basePath,"theories/example_cutlery_and_fruit/schemaInterpretation.dfl")
    updQTheory = os.path.join(basePath,"theories/example_cutlery_and_fruit/updateQueries.dfl")
    interpMasksTheory = os.path.join(basePath,"theories/example_cutlery_and_fruit/interpretMasks.dfl")
    storeMasksTheory = os.path.join(basePath,"theories/example_cutlery_and_fruit/storeMasks.dfl")
    backgroundFacts = os.path.join(basePath,"theories/example_cutlery_and_fruit/backgroundFacts.dfl")
    defaultFacts = os.path.join(basePath,"theories/example_cutlery_and_fruit/defaultFacts.dfl")

    # IMPORTANT: this will be our registry of "wires", connections between khafre subprocesses.
    # Keep this variable alive at least as long as the subprocesses are running.

    wireList={}

    # For ease of start up, cleanup, and setting up termination handlers, process objects should be stored in a dictionary.
    
    procs = {}

    # We will need a debug visualizer, a depth estimator, and object detection processes.
    # Construct process objects. These are not started yet.
        
    procs["vc"] = RecordedVideoFeed()
    if not headless:
        procs["dbgP"] = DbgVisualizer()
    procs["objP"] = ByteTracker()
    # Monocular depth estimation is VERY computationally expensive, try to have it run on the GPU
    procs["dptP"] = TransformerDepthSegmentationWrapper(device="cuda")
    procs["conP"] = ContactDetection()
    procs["optP"] = OpticalFlow()
    procs["decP"] = Decluder()
    procs["reasoner"] = Reasoner()
    procs["storage"] = YOLOFrameSaver()

    # Set up the connections to dbg visualizer and object detection.
    if not headless:
        drawWire("Video Dbg", ("DbgImg", procs["vc"]), [("Video", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Dbg Depth", ("DbgImg", procs["dptP"]), [("Depth Estimation", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Dbg Obj Seg", ("DbgImg", procs["objP"]), [("Object Detection/Segmentation", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Dbg Contact", ("DbgImg", procs["conP"]), [("Contact Detection", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Dbg Optical Flow", ("DbgImg", procs["optP"]), [("Optical Flow (sparse)", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Dbg Storage", ("DbgImg", procs["storage"]), [("Frame Storage", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Dbg Declude", ("DbgImg", procs["decP"]), [("Declusion", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)

    drawWire("Input Image", ("OutImg", procs["vc"]), [("InpImg", procs["objP"]), ("InpImg", procs["dptP"]), ("InpImg", procs["decP"]), ("InpImg", procs["optP"]), ("InpImg", procs["reasoner"])], (imgHeight, imgWidth, 3), numpy.uint8, wireList=wireList)
    drawWire("Mask Image", ("OutImg", procs["objP"]), [("MaskImg", procs["decP"]), ("MaskImg", procs["conP"]), ("MaskImg", procs["optP"]), ("MaskImg", procs["reasoner"])], (imgHeight, imgWidth), numpy.uint16, wireList=wireList)
    drawWire("Depth Image", ("OutImg", procs["dptP"]), [("DepthImg", procs["conP"]), ("DepthImg", procs["optP"])], (imgHeight, imgWidth), numpy.float32, wireList=wireList)
    drawWire("OutImg Contact", ("OutImg", procs["conP"]), [("ContactMask", procs["reasoner"])], (imgHeight, imgWidth), numpy.uint32, wireList=wireList)
    drawWire("OutImg OpticalFlow", ("OutImg", procs["optP"]), [("Optical Flow (sparse)", procs["reasoner"])], (imgHeight, imgWidth), numpy.uint32, wireList=wireList)
    drawWire("OutImg Declusion", ("OutImg", procs["decP"]), [("DeclusionMask", procs["reasoner"])], (imgHeight, imgWidth), numpy.uint32, wireList=wireList)
    drawWire("Masks to Store", ("OutImg", procs["reasoner"]), [("InpImg", procs["storage"])], (imgHeight, imgWidth, 3), numpy.uint8, wireList=wireList)
        
    procs["reasoner"].registerWorker("contact", procs["conP"])
    procs["reasoner"].registerWorker("opticalFlow", procs["optP"])
    procs["reasoner"].registerWorker("declude", procs["decP"])

    # Optional, but STRONGLY recommended: set up signal handlers. The handlers will trigger the 
    # termination of the various subprocesses. Alternatively, ensure in some other way that
    # subprocesses are terminated at exit.
    # Note that the previously registered sigint and sigterm handlers are returned, so you can
    # restore them if you need to. That may be the case when you stop the khafre processes manually,
    # and then wish to continue running your program anyway.

    sigintHandler, sigtermHandler = setSignalHandlers(procs)

    # Only after all connection objects -- shared memories and notification queues -- are set up, we can start.

    startKhafreProcesses(procs)
        
    # RECOMMENDED: tell the object detection to load a model AFTER starting the process. This might avoid some unnecessary
    # copying of a large object when starting the process.
        
    procs["objP"].sendCommand(("LOAD", ("yolov8x-seg.pt",)))
    #procs["objP"].sendCommand(("LOAD", ("yolo11x-seg.pt",)))
    procs["objP"].sendCommand(("START", ()))
    procs["dptP"].sendCommand(("LOAD", ("vinvino02/glpn-nyu",)))

    procs["reasoner"].sendCommand(("LOAD_THEORY", ("perception interpretation", percIntTheory)))
    procs["reasoner"].sendCommand(("LOAD_THEORY", ("update schemas", updSchTheory)))
    procs["reasoner"].sendCommand(("LOAD_THEORY", ("connectivity queries", connQTheory)))
    procs["reasoner"].sendCommand(("LOAD_THEORY", ("schema closure", schClsTheory)))
    procs["reasoner"].sendCommand(("LOAD_THEORY", ("schema interpretation", schIntTheory)))
    procs["reasoner"].sendCommand(("LOAD_THEORY", ("update questions", updQTheory)))
    procs["reasoner"].sendCommand(("LOAD_THEORY", ("interpret masks", interpMasksTheory)))
    procs["reasoner"].sendCommand(("LOAD_THEORY", ("store masks", storeMasksTheory)))
    procs["reasoner"].sendCommand(("LOAD_FACTS", (backgroundFacts,)))
    procs["reasoner"].sendCommand(("REGISTER_STORAGE_DESTINATION", ("InpImg", "OutImg")))

    procs["reasoner"].sendCommand(("TRIGGER", (defaultFacts,)))
    procs["reasoner"].sendCommand(("SET_PATH", (eventPath,)))
    procs["storage"].sendCommand(("SET_PATH", (storagePath,)))

    procs["vc"].sendCommand(("LOAD", (arguments.input_video,)))
        
    procs["vc"].waitForLoad()

    # Define and run some code that actually does something with the set up processes.

    if not headless:                
        print("Starting object segmentation and depth estimation will take a while, wait a few seconds for debug windows for them to show up.\nPress ESC to exit. (By the way, this is process %s)" % str(os.getpid()))

        # This function will be called repeatedly until some condition happens: either a key is released,
        # or something inside the function triggers the end.
            
        # This function sleeps while the stored video process produces frames, and will return false when
        # there are no more frames so that the looping is stopped.

        def exampleFn(procs):
            time.sleep(0.1)
            return not procs["vc"].hasEnded()

        # Loop the above function until a key is released. For this example, that will be the ESCAPE key.

        repeatUntilKey(lambda : exampleFn(procs))

    else:
        print("Starting khafre in headless mode.\nYou may stop this process and its children by a keyboard interrupt, or by sending this process a terminate signal.\nBy the way, this is process %s" % str(os.getpid()))

        # Loop while there are frames left to process.

        procs["vc"].waitForEnd()

    # Once the video is finished, tell reasoning to also make a summary.html of the selected frames.

    procs["reasoner"].sendCommand(("STORE_SUMMARY", ()))
    time.sleep(0.1)

    # A clean exit: stop all subprocesses.
    # In general, you can use stopKhafreProcesses to do what it says. Note that by default it will not raise exceptions.
    # If you want to stop processes and handle exceptions yourself, run stopKhafreProcesses(procs, exceptions=True)

    stopKhafreProcesses(procs)

if "__main__" == __name__:
    main()

