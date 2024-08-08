import argparse
import base64
import cv2 as cv
import numpy
import os
from PIL import Image
from pynput.keyboard import Key, Listener
import signal
import sys
import time

from khafre.bricks import SHMPort, RatedSimpleQueue, drawWire, setSignalHandlers, startKhafreProcesses, stopKhafreProcesses
from khafre.dbgvis import DbgVisualizer
from khafre.segmentation import YOLOObjectSegmentationWrapper
from khafre.videocapture import RecordedVideoFeed


## DbgVis example: shows how to set up a connection to the khafre debug visualizer.
# In this example, what we will visualize is a resized copy of one of your monitors.
# The main process -- this one -- will capture the screen, rescale it, and send it
# to a subprocess: the khafre debug visualizer.
# The CPU usage of this process will be 100%: it will send images as fast as it can.
# The debug visualizer meanwhile doesn't need to do much and its CPU usage will be
# much lower.

# Auxiliary objects to exit on key press (or rather, release)
goOn={"goOn":True}

def on_press(key):
    pass

def on_release(key):
    if key == Key.esc:
        goOn["goOn"] = False
        # Stop listener
        return False

## An auxiliary function to set up a signal handler for SIGTERM and SIGINT
def doExit(signum, frame, dbgP, vc, objP):
    dbgP.stop()
    vc.stop()
    objP.stop()
    sys.exit()

def main():
    
    parser = argparse.ArgumentParser(prog='stored_video_input', description='Analyze a video file using khafre', epilog='')
    parser.add_argument('-iv', '--input_video', default="", help="Path to the video file that will be used as input for khafre.")
    
    arguments = parser.parse_args()
    if ("" == arguments.input_video) or (not os.path.isfile(arguments.input_video)):
        print("Must provide an input file!")
        sys.exit(0)

    # IMPORTANT: this will be our registry of "wires", connections between khafre subprocesses.
    # Keep this variable alive at least as long as the subprocesses are running.

    wireList = {}

    # For ease of start up, cleanup, and setting up termination handlers, process objects should be stored in a dictionary.
    
    procs = {}

    imgWidth,imgHeight = (int(1280*480/720),480)
        
    procs["dbgP"] = DbgVisualizer()
    procs["vc"] = RecordedVideoFeed()
    procs["objP"] = YOLOObjectSegmentationWrapper()

    drawWire("Video Dbg", [("DbgImg", procs["vc"])], [("Video", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, RatedSimpleQueue, wireList=wireList)
    drawWire("Video Img", [("OutImg", procs["vc"])], [("InpImg", procs["objP"])], (imgHeight, imgWidth, 3), numpy.uint8, RatedSimpleQueue, wireList=wireList)
    drawWire("Dbg Obj Seg", [("DbgImg", procs["objP"])], [("Object Detection/Segmentation", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, RatedSimpleQueue, wireList=wireList)

    # Optional, but STRONGLY recommended: set up signal handlers. The handlers will trigger the 
    # termination of the various subprocesses. Alternatively, ensure in some other way that
    # subprocesses are terminated at exit.
    # Note that the previously registered sigint and sigterm handlers are returned, so you can
    # restore them if you need to. That may be the case when you stop the khafre processes manually,
    # and then wish to continue running your program anyway.

    sigintHandler, sigtermHandler = setSignalHandlers(procs)
    
    # Only after all connection objects -- shared memories and notification queues -- are set up, we can start.

    startKhafreProcesses(procs)
        
    procs["vc"].sendCommand(("LOAD", (arguments.input_video,)))
    procs["objP"].sendCommand(("LOAD", ("yolov8x-seg.pt",)))

    # Finally, you can enter a loop in which the screen is captured and sent to the DbgVisualizer.
    # You will also see how many fps your system can manage with this code.
    # Just as an example, setting up a way to exit the loop without signals as well -- in this case,
    # when releasing the ESC key.

    print("Press ESC to exit. (By the way, this is process %s)" % str(os.getpid()))
    with Listener(on_press=on_press, on_release=on_release) as listener:
        while goOn["goOn"]:

            # Place the image into the shared port. Also, notify the consumer (DbgVis) that something happened.
            # Note: screenshot is likely larger than the producer's image. producer.send will automatically resize
            # in this case.

            # DbgVis can also print something for us. We could also have sent a notification with an empty string
            # instead.
            procs["vc"].sendCommand(("FRAME", ()))
            time.sleep(0.1)
            if procs["vc"].hasEnded():
                goOn["goOn"] = False
                listener.stop()
            # Usually, some waiting time between iterations of such a loop would also be needed. However, usually
            # the kind of processes that generate images, such as screenshots and resizes, are "slow", and can
            # function as a delay themselves.

        listener.join()

    # A clean exit: stop all subprocesses.
    # In general, you can use stopKhafreProcesses to do what it says. Note that by default it will not raise exceptions.
    # If you want to stop processes and handle exceptions yourself, run stopKhafreProcesses(procs, exceptions=True)

    stopKhafreProcesses(procs)

if "__main__" == __name__:
    main()

