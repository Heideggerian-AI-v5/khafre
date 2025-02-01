import argparse
import cv2 as cv
import numpy
import os
import sys
import time

from khafre.bricks import drawWire, setSignalHandlers, startKhafreProcesses, stopKhafreProcesses
from khafre.dbgvis import DbgVisualizer
from khafre.depth import TransformerDepthSegmentationWrapper
from khafre.flowtracker import FlowTracker
from khafre.videocapture import RecordedVideoFeed


### Trying out a custom tracker.

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
    parser = argparse.ArgumentParser(prog='stored_video_input', description='Analyze a video file using khafre', epilog='')
    parser.add_argument('-iv', '--input_video', default="", help="REQUIRED: Path to the video file that will be used as input for khafre.")
    #parser.add_argument('-nogui', '--nogui', action='store_true', help='OPTIONAL: Flag to disable debug visualization. Will also disable early termination via ESCAPE keypress (early termination via keyboard interrupt or terminate signal remains available). Use this on systems with no window manager. You will likely need this when running khafre on remote machines too.')
    parser.add_argument('-ml', '--max_lines', default="640", help="OPTIONAL: Maximum number of lines for a video frame. If the input video has higher resolution, it will be downscaled so that its frame height matches this parameter. Default is 640. If parameter given, must be convertible to integer.")
    
    arguments = parser.parse_args()
    
    headless = False #arguments.nogui
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

    ######### Part 1: khafre usage example.

    # IMPORTANT: this will be our registry of "wires", connections between khafre subprocesses.
    # Keep this variable alive at least as long as the subprocesses are running.

    wireList={}

    # For ease of start up, cleanup, and setting up termination handlers, process objects should be stored in a dictionary.
    
    procs = {}

    # We will need an input and (unless running headless) a debug visualizer, and several other processes:
    #     object tracking
    #     [depth estimation]
    # Construct process objects. These are not started yet.
    
    procs["vc"] = RecordedVideoFeed()
    if not headless:
        procs["dbgP"] = DbgVisualizer()
    procs["objP"] = FlowTracker()
    # Monocular depth estimation is VERY computationally expensive, try to have it run on the GPU
    #procs["dptP"] = TransformerDepthSegmentationWrapper(device="cuda")

    # Set up the connections between processes.

    if not headless:
        drawWire("Video Dbg", ("DbgImg", procs["vc"]), [("Video", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Dbg Obj Seg", ("DbgImg", procs["objP"]), [("Object Detection/Segmentation", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        #drawWire("Dbg Depth", ("DbgImg", procs["dptP"]), [("Depth Estimation", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)

    drawWire("Input Image", ("OutImg", procs["vc"]), [("InpImg", procs["objP"])], (imgHeight, imgWidth, 3), numpy.uint8, wireList=wireList)
    #drawWire("Input Image", ("OutImg", procs["vc"]), [("InpImg", procs["objP"]), ("InpImg", procs["dptP"])], (imgHeight, imgWidth, 3), numpy.uint8, wireList=wireList)
    #drawWire("Depth Image", ("OutImg", procs["dptP"]), [("DepthImg", procs["objP"])], (imgHeight, imgWidth), numpy.float32, wireList=wireList)

    # Optional, but STRONGLY recommended: set up signal handlers. The handlers will trigger the 
    # termination of the various subprocesses. Alternatively, ensure in some other way that
    # subprocesses are terminated at exit.
    # Note that the previously registered sigint and sigterm handlers are returned, so you can
    # restore them if you need to. That may be the case when you stop the khafre processes manually,
    # and then wish to continue running your program anyway.

    sigintHandler, sigtermHandler = setSignalHandlers(procs)

    # Only after all connection objects -- shared memories and notification queues -- are set up, we can start.

    startKhafreProcesses(procs)

    # Load neural network models for object detection/tracking and depth estimation.
    # RECOMMENDED: tell the object detection to load a model AFTER starting the process. This might avoid some unnecessary
    # copying of a large object when starting the process.

    #procs["dptP"].sendCommand(("LOAD", ("vinvino02/glpn-nyu",)))
    procs["objP"].sendCommand(("LOAD", ("yolov8x-seg.pt",)))
    procs["objP"].sendCommand(("START", ()))

    # Load and start sending the video to khafre.
    procs["vc"].sendCommand(("LOAD", (inputVideo,)))

    # Wait for the video to load.
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

    # A clean exit: stop all subprocesses.
    # In general, you can use stopKhafreProcesses to do what it says. Note that by default it will not raise exceptions.
    # If you want to stop processes and handle exceptions yourself, run stopKhafreProcesses(procs, exceptions=True)

    stopKhafreProcesses(procs)

if "__main__" == __name__:
    main()

