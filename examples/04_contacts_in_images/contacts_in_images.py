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
from khafre.tracker import ByteTracker, SegmentAnythingWrapper
from khafre.contact import ContactDetection
from khafre.optical_flow import OpticalFlow
from khafre.reasoning import Reasoner
from khafre.framestore import YOLOFrameSaver
from khafre.imagesequence import ImageSequence, imgDimPicker
from khafre.storage import StoreTriples
from khafre.storage import StoreMasks


### Contacts in images example:
# use SAM to look for particular object types in a collection of images. The images do not form a sequence.
# Also check for contacts between the objects.


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
    parser = argparse.ArgumentParser(prog='contacts_in_images', description='Check for contacts between given object types.', epilog='')
    parser.add_argument('-ii', '--input_images', default="", help="REQUIRED: Path to a folder containing a collection of individual images.")
    parser.add_argument('-nogui', '--nogui', action='store_true', help='OPTIONAL: Flag to disable debug visualization. Will also disable early termination via ESCAPE keypress (early termination via keyboard interrupt or terminate signal remains available). Use this on systems with no window manager. You will likely need this when running khafre on remote machines too.')
    parser.add_argument("-oc", '--object_classes', default="", help="REQUIRED: Path to a text file where each line is an object class name. Class names should contain no spaces -- use underscores instead.")
    parser.add_argument('-tp', '--triples_path', default="./triples", help="OPTIONAL: Path to the folder where khafre will store triples about found objects, their contact relations, and the object masks.")
    parser.add_argument('-ml', '--max_lines', default="640", help="OPTIONAL: Maximum number of lines for a video frame. If the input video has higher resolution, it will be downscaled so that its frame height matches this parameter. Default is 640. If parameter given, must be convertible to integer.")
    
    arguments = parser.parse_args()
    
    headless = arguments.nogui
    if not headless:
        from khafre.utils import repeatUntilKey
    
    try:
        objectClasses = open(arguments.object_classes).read().splitlines()
        objectClasses = [x.strip().replace(" ", "_") for x in objectClasses]
        if 0 == len(objectClasses):
            raise ValueError("Must have at least one object class to track.")
    except Exception as e:
        print("Encountered exception while reading the object classes.")
        print(e)
        print("Will now exit.")
        sys.exit(0)
        
    try:
        maxLines = int(arguments.max_lines)
    except Exception as e:
        print("Encountered exception while reading the max lines parameter.")
        print(e)
        print("Will now exit.")
        sys.exit(0)
        
    try:
        inputImageFolder = os.path.abspath(arguments.input_images).replace("\\","/")
        if ("" == inputImageFolder) or (not os.path.isdir(inputImageFolder)):
            raise ValueError("Provided input image folder is not a valid or readable folder.")
        imgHeight, imgWidth = imgDimPicker(inputImageFolder)
    except Exception as e:
        print("Must provide a valid input image folder!")
        print(e)
        print("Will now exit.")
        sys.exit(0)
    
    triplesPath = os.path.abspath(arguments.triples_path).replace("\\","/")
    
    if not os.path.isdir(triplesPath):
        try:
            os.mkdir(triplesPath)
        except Exception as e:
            print("Encountered exception during an attempt to create a folder for the triples files.")
            print(e)
            print("Will now exit.")
            sys.exit(0)

    # IMPORTANT: this will be our registry of "wires", connections between khafre subprocesses.
    # Keep this variable alive at least as long as the subprocesses are running.

    wireList={}

    # For ease of start up, cleanup, and setting up termination handlers, process objects should be stored in a dictionary.
    
    procs = {}

    # We will need a debug visualizer, a depth estimator, and object detection processes.
    # Construct process objects. These are not started yet.

    procs["is"] = ImageSequence()
    if not headless:
        procs["dbgP"] = DbgVisualizer()
    procs["objP"] = ByteTracker(segmenter=SegmentAnythingWrapper())
    # Monocular depth estimation is VERY computationally expensive, try to have it run on the GPU
    procs["dptP"] = TransformerDepthSegmentationWrapper(device="cuda")
    procs["conP"] = ContactDetection()
    procs["sttP"] = StoreTriples({"isA":"rdf:type"})
    procs["sttcP"] = StoreTriples({"contact": "affordances_situations:contact"})
    procs["stmP"] = StoreMasks()

    # Set up the connections to dbg visualizer and object detection.
    if not headless:
        drawWire("Video Dbg", ("DbgImg", procs["is"]), [("Image", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Dbg Depth", ("DbgImg", procs["dptP"]), [("Depth Estimation", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Dbg Obj Seg", ("DbgImg", procs["objP"]), [("Object Detection/Segmentation", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Dbg Contact", ("DbgImg", procs["conP"]), [("Contact Detection", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)

    drawWire("Input Image", ("OutImg", procs["is"]), [("InpImg", procs["objP"]), ("InpImg", procs["dptP"])], (imgHeight, imgWidth, 3), numpy.uint8, wireList=wireList)
    drawWire("Mask Image", ("OutImg", procs["objP"]), [("MaskImg", procs["conP"]), ("InpImg", procs["sttP"]), ("InpImg", procs["stmP"])], (imgHeight, imgWidth), numpy.uint16, wireList=wireList)
    drawWire("Depth Image", ("OutImg", procs["dptP"]), [("DepthImg", procs["conP"])], (imgHeight, imgWidth), numpy.float32, wireList=wireList)
    drawWire("Contact triples", ("OutImg", procs["conP"]), [("InpImg", procs["sttcP"])], (imgHeight, imgWidth), numpy.uint32, wireList=wireList)
    
    """
    if not headless:
        drawWire("Video Dbg", ("DbgImg", procs["is"]), [("Image", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Dbg Obj Seg", ("DbgImg", procs["objP"]), [("Object Detection/Segmentation", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)

    drawWire("Input Image", ("OutImg", procs["is"]), [("InpImg", procs["objP"])], (imgHeight, imgWidth, 3), numpy.uint8, wireList=wireList)
    drawWire("Object triples", ("OutImg", procs["objP"]), [("InpImg", procs["sttP"]), ("InpImg", procs["stmP"])], (imgHeight, imgWidth), numpy.uint32, wireList=wireList)
    """

    # A bit of a hack, but we don't need to adjust queries via reasoning so we may as well hardcode them into the bricks:
    procs["objP"]._queries = [("find", x, None) for x in objectClasses]
    procs["conP"]._queries = [("contact/query", x, None) for x in objectClasses]


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
        
    procs["objP"].sendCommand(("LOAD", ("IDEA-Research/grounding-dino-tiny", "facebook/sam2.1-hiera-tiny")))
    procs["objP"].sendCommand(("START", ()))
    procs["dptP"].sendCommand(("LOAD", ("depth-anything/Depth-Anything-V2-Small-hf",)))
    procs["sttP"].sendCommand(("SET_PATH", (triplesPath,)))
    procs["sttP"].sendCommand(("SET_COLLECTION", ("isA",)))
    procs["sttcP"].sendCommand(("SET_PATH", (triplesPath,)))
    procs["sttcP"].sendCommand(("SET_COLLECTION", ("contact",)))
    procs["stmP"].sendCommand(("SET_PATH", (triplesPath,)))

    procs["is"].sendCommand(("LOAD", (inputImageFolder,)))
        

    # Define and run some code that actually does something with the set up processes.

    if not headless:                
        print("Starting object segmentation and depth estimation will take a while, wait a few seconds for debug windows for them to show up.\nPress ESC to exit. (By the way, this is process %s)" % str(os.getpid()))

        # This function will be called repeatedly until some condition happens: either a key is released,
        # or something inside the function triggers the end.
            
        # This function sleeps while the stored video process produces frames, and will return false when
        # there are no more frames so that the looping is stopped.

        def exampleFn(procs):
            time.sleep(0.1)
            return not procs["is"].hasEnded()

        # Loop the above function until a key is released. For this example, that will be the ESCAPE key.

        repeatUntilKey(lambda : exampleFn(procs))

    else:
        print("Starting khafre in headless mode.\nYou may stop this process and its children by a keyboard interrupt, or by sending this process a terminate signal.\nBy the way, this is process %s" % str(os.getpid()))

        # Loop while there are frames left to process.

        procs["is"].waitForEnd()

    # A clean exit: stop all subprocesses.
    # In general, you can use stopKhafreProcesses to do what it says. Note that by default it will not raise exceptions.
    # If you want to stop processes and handle exceptions yourself, run stopKhafreProcesses(procs, exceptions=True)

    stopKhafreProcesses(procs)

if "__main__" == __name__:
    main()

