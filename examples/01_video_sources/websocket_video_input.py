import argparse
import cv2 as cv
import numpy
import os
import sys
import time

from khafre.bricks import drawWire, setSignalHandlers, startKhafreProcesses, stopKhafreProcesses
from khafre.dbgvis import DbgVisualizer
from khafre.segmentation import YOLOObjectSegmentationWrapper
from khafre.videocapture import WebStream
from khafre.utils import repeatUntilKey


## Websocket video example: shows how to set up a connection from a websocket offering a video stream to the khafre debug visualizer and object segmentation.

def main():
    
    parser = argparse.ArgumentParser(prog='socket_video_input', description='Analyze a video stream published by a socket using khafre', epilog='')
    parser.add_argument('-uri', '--uri', default="", help="URI of the video server.")
    
    arguments = parser.parse_args()
    if ("" == arguments.uri):
        print("Must supply a URI")
        sys.exit(0)

    # IMPORTANT: this will be our registry of "wires", connections between khafre subprocesses.
    # Keep this variable alive at least as long as the subprocesses are running.

    wireList = {}

    # For ease of start up, cleanup, and setting up termination handlers, process objects should be stored in a dictionary.
    
    procs = {}

    # Frames will be resized where necessary.
    imgWidth,imgHeight = (400,300)
        
    procs["dbgP"] = DbgVisualizer()
    procs["vc"] = WebStream()
    procs["objP"] = YOLOObjectSegmentationWrapper()

    drawWire("Video Dbg", ("DbgImg", procs["vc"]), [("Video", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
    drawWire("Video Img", ("OutImg", procs["vc"]), [("InpImg", procs["objP"])], (imgHeight, imgWidth, 3), numpy.uint8, wireList=wireList)
    drawWire("Dbg Obj Seg", ("DbgImg", procs["objP"]), [("Object Detection/Segmentation", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)

    # Optional, but STRONGLY recommended: set up signal handlers. The handlers will trigger the 
    # termination of the various subprocesses. Alternatively, ensure in some other way that
    # subprocesses are terminated at exit.
    # Note that the previously registered sigint and sigterm handlers are returned, so you can
    # restore them if you need to. That may be the case when you stop the khafre processes manually,
    # and then wish to continue running your program anyway.

    sigintHandler, sigtermHandler = setSignalHandlers(procs)
    
    # Only after all connection objects -- shared memories and notification queues -- are set up, we can start.

    startKhafreProcesses(procs)
        
    procs["vc"].sendCommand(("CONNECT", (arguments.uri,)))
    procs["objP"].sendCommand(("LOAD", ("yolov8x-seg.pt",)))
    
    # Define and run some code that actually does something with the set up processes.

    # This function will be called repeatedly until some condition happens: either a key is released,
    # or something inside the function triggers the end.
        
    def exampleFn(procs):
        time.sleep(0.1)
        return True

    print("Press ESC to exit. (By the way, this is process %s)" % str(os.getpid()))
    
    # Loop the above function until a key is released. For this example, that will be the ESCAPE key.
    
    repeatUntilKey(lambda : exampleFn(procs))

    # A clean exit: stop all subprocesses.
    # In general, you can use stopKhafreProcesses to do what it says. Note that by default it will not raise exceptions.
    # If you want to stop processes and handle exceptions yourself, run stopKhafreProcesses(procs, exceptions=True)

    stopKhafreProcesses(procs)

if "__main__" == __name__:
    main()

