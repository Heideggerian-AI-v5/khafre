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

from khafre.bricks import RatedSimpleQueue, SHMPort, drawWire, setSignalHandlers, startKhafreProcesses, stopKhafreProcesses
from khafre.dbgvis import DbgVisualizer
from khafre.depth import TransformerDepthSegmentationWrapper
from khafre.segmentation import YOLOObjectSegmentationWrapper
from khafre.contact import ContactDetection
from khafre.optical_flow import OpticalFlow

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

    # For ease of start up, cleanup, and setting up termination handlers, process objects should be stored in a dictionary.
    
    procs = {}

    with mss.mss() as sct:

        monitor = sct.monitors[1]

        imgWidth,imgHeight = (int(monitor["width"]*240.0/monitor["height"]), 240)
        
        # We will need a debug visualizer, a depth estimator, and object detection processes.
        # Construct process objects. These are not started yet.
        
        procs["dbgP"] = DbgVisualizer()
        procs["objP"] = YOLOObjectSegmentationWrapper()
        # Monocular depth estimation is VERY computationally expensive, try to have it run on the GPU
        procs["dptP"] = TransformerDepthSegmentationWrapper(device="cuda")
        procs["conP"] = ContactDetection()
        procs["optP"] = OpticalFlow()

        # Set up the connections to dbg visualizer and object detection.

        drawWire("Screenshot Cam", [], [("Screenshot Cam", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, RatedSimpleQueue, wireList=wireList)
        drawWire("Input Image", [], [("InpImg", procs["objP"]), ("InpImg", procs["dptP"]), ("InpImg", procs["optP"])], (imgHeight, imgWidth, 3), numpy.uint8, RatedSimpleQueue, wireList=wireList)
        drawWire("Dbg Obj Seg", [("DbgImg", procs["objP"])], [("Object Detection/Segmentation", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, RatedSimpleQueue, wireList=wireList)
        drawWire("Dbg Depth", [("DbgImg", procs["dptP"])], [("Depth Estimation", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, RatedSimpleQueue, wireList=wireList)
        drawWire("Mask Image", [("OutImg", procs["objP"])], [("MaskImg", procs["conP"]), ("MaskImg", procs["optP"])], (imgHeight, imgWidth), numpy.uint16, RatedSimpleQueue, wireList=wireList)
        drawWire("Depth Image", [("OutImg", procs["dptP"])], [("DepthImg", procs["conP"]), ("DepthImg", procs["optP"])], (imgHeight, imgWidth), numpy.float32, RatedSimpleQueue, wireList=wireList)
        drawWire("Dbg Contact", [("DbgImg", procs["conP"])], [("Contact Detection", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, RatedSimpleQueue, wireList=wireList)
        drawWire("Dbg Optical Flow", [("DbgImg", procs["optP"])], [("Optical Flow (sparse)", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, RatedSimpleQueue, wireList=wireList)
        
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
        
        procs["objP"].sendCommand(("LOAD", ("yolov8n-seg.pt",)))
        procs["dptP"].sendCommand(("LOAD", ("vinvino02/glpn-nyu",)))

        procs["conP"].getGoalQueue().put([("contact/query", "cup", "table"), ("contact/query", "cup", "dining table")])
        procs["optP"].getGoalQueue().put([("opticalFlow/query/relativeMovement", "cup", "table"), ("opticalFlow/query/relativeMovement", "cup", "dining table")])

        print("Starting object segmentation and depth estimation will take a while, wait a few seconds for debug windows for them to show up.\nPress ESC to exit. (By the way, this is process %s)" % str(os.getpid()))
        with Listener(on_press=on_press, on_release=on_release) as listener:
            while goOn["goOn"]:

                screenshot = getImg(sct, monitor)

                # Send the screenshot to dbgvis and object detection.
                
                wireList["Input Image"].publish((screenshot*255).astype(numpy.uint8), {"imgId": str(time.perf_counter())})
                wireList["Screenshot Cam"].publish(screenshot, "")
                
            listener.join()

        # A clean exit: stop all subprocesses.
        # In general, you can use stopKhafreProcesses to do what it says. Note that by default it will not raise exceptions.
        # If you want to stop processes and handle exceptions yourself, run stopKhafreProcesses(procs, exceptions=True)

        stopKhafreProcesses(procs)

if "__main__" == __name__:
    main()

