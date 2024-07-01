import base64
import cv2 as cv
import numpy
import mss
from multiprocessing import Queue
import os
from PIL import Image
from pynput.keyboard import Key, Listener
import signal
import sys
import time

from khafre.bricks import RatedSimpleQueue, SHMPort, drawWire
from khafre.dbgvis import DbgVisualizer
from khafre.nnwrappers import YOLOObjectSegmentationWrapper

### Object detection/segmentation example: shows how to set up a connection to the khafre object detection,
# and between it and a debug visualizer. See the dbgvis_setup_input_connection.py example for more comments
# on the debug visualizer.
# In this example, we will attempt to have a pretrained YOLO model recognize objects visible on screen.

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
def doExit(signum, frame, dbgP, objP):
    dbgP.stop()
    objP.stop()
    sys.exit()

# Define a function to capture the image from the screen.
def getImg(sct, monitor):
    sct_img = sct.grab(monitor)
    pilImage = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    rawImage = numpy.ascontiguousarray(numpy.float32(numpy.array(pilImage)[:,:,(2,1,0)]))/255.0
    return rawImage

def main():

    # IMPORTANT: this will be our registry of "wires", connections between khafre subprocesses.
    # Keep this variable alive at least as long as the subprocesses are running.

    wireList={}

    with mss.mss() as sct:

        monitor = sct.monitors[1]

        imgWidth,imgHeight = (int(monitor["width"]*240.0/monitor["height"]), 240)
        
        # We will need a debug visualizer and object detection processes.        
        # Construct process objects. These are not started yet.
        
        dbgP = DbgVisualizer()
        objP = YOLOObjectSegmentationWrapper()

        # Set up the connections to dbg visualizer and object detection.

        # Normally there should be a wire for the OutImg connection of the object detection/segmentation process, 
        # however we will not use that result in this example and only visualize results via debug.

        drawWire("Screenshot Cam", [], [("Screenshot Cam", dbgP)], (imgHeight, imgWidth, 3), numpy.float32, RatedSimpleQueue, wireList=wireList)
        drawWire("Input Image", [], [("InpImg", objP)], (imgHeight, imgWidth, 3), numpy.uint8, RatedSimpleQueue, wireList=wireList)
        drawWire("Dbg Obj Seg", [("DbgImg", objP)], [("Object Detection/Segmentation", dbgP)], (imgHeight, imgWidth, 3), numpy.float32, RatedSimpleQueue, wireList=wireList)
        
        # Optional, but STRONGLY recommended: set signal handlers that will ensure the subprocesses terminate on exit.
        signal.signal(signal.SIGTERM, lambda signum, frame: doExit(signum, frame, dbgP, objP))
        signal.signal(signal.SIGINT, lambda signum, frame: doExit(signum, frame, dbgP, objP))

        # Only after all connection objects -- shared memories and notification queues -- are set up, we can start.
        dbgP.start()
        objP.start()
        
        # RECOMMENDED: tell the object detection to load a model AFTER starting the process. This might avoid some unnecessary
        # copying of a large object when starting the process.
        
        objP.sendCommand(("LOAD", ("yolov8n-seg.pt",)))

        print("Starting object segmentation will take a while, wait a few seconds for a debug window labeled \"Object Detection/Segmentation\".\nPress ESC to exit. (By the way, this is process %s)" % str(os.getpid()))
        with Listener(on_press=on_press, on_release=on_release) as listener:
            while goOn["goOn"]:

                screenshot = getImg(sct, monitor)

                # Send the screenshot to dbgvis and object detection.
                
                wireList["Input Image"].publish((screenshot*255).astype(numpy.uint8), True)
                wireList["Screenshot Cam"].publish(screenshot, "")
                
            listener.join()

        # A clean exit: stop all subprocesses.
        
        objP.stop()
        dbgP.stop()

if "__main__" == __name__:
    main()

