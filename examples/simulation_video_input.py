import base64
import cv2 as cv
import numpy
import os
from PIL import Image
from pynput.keyboard import Key, Listener
import signal
import sys
import time

from khafre.bricks import drawWire, setSignalHandlers, startKhafreProcesses, stopKhafreProcesses
from khafre.dbgvis import DbgVisualizer
#from khafre.depth import TransformerDepthSegmentationWrapper
from khafre.segmentation import YOLOObjectSegmentationWrapper
#from khafre.contact import ContactDetection
#from khafre.optical_flow import OpticalFlow
from khafre.sapien_wrapper import SapienSim
from khafre.utils import repeatUntilKey

### Object detection/segmentation example: shows how to set up a connection to the khafre object detection,
# and between it and a debug visualizer. See the dbgvis_setup_input_connection.py example for more comments
# on the debug visualizer.
# In this example, we will attempt to have a pretrained YOLO model recognize objects visible on screen.

def main():

    asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../assets")

    # IMPORTANT: this will be our registry of "wires", connections between khafre subprocesses.
    # Keep this variable alive at least as long as the subprocesses are running.

    wireList={}

    # For ease of start up, cleanup, and setting up termination handlers, process objects should be stored in a dictionary.
    
    procs = {}

    imgWidth,imgHeight = 640,480

    procs["sim"] = SapienSim(height=imgHeight,width=imgWidth,viewer=False)
    procs["dbgP"] = DbgVisualizer()
    procs["objP"] = YOLOObjectSegmentationWrapper()

    # Set up the connections to dbg visualizer and object detection.

    drawWire("Simulation Cam", ("DbgImg", procs["sim"]), [("Simulation Cam", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
    drawWire("Input Image", ("OutImg", procs["sim"]), [("InpImg", procs["objP"])], (imgHeight, imgWidth, 3), numpy.uint8, wireList=wireList)
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

    procs["objP"].sendCommand(("LOAD", ("yolov8n-seg.pt",)))
    procs["sim"].sendCommand(["SET AMBIENT LIGHT",[[0.5, 0.5, 0.5]]])
    procs["sim"].sendCommand(["ADD DIRECTIONAL LIGHT",[[0, 1, -1], [0.5, 0.5, 0.5]]])
    procs["sim"].sendCommand(["LOAD ASSET", ["table", os.path.join(asset_path, "table/table.urdf"), [0, 0, 0.44], [1, 0, 0, 0]]])
    procs["sim"].sendCommand(["LOAD ACTOR", ["mug", os.path.join(asset_path, "beermug/BeerMugCollision.obj"), os.path.join(asset_path, "beermug/BeerMugVisual.obj"), [-0.2, 0, 0.44 + 0.05], [1, 0, 0, 0]]])
    procs["sim"].sendCommand(["SET CAMERA POSE", [numpy.array([-2, -2, 3])]])

    # Define and run some code that actually does something with the set up processes.

    # This function will be called repeatedly until some condition happens: either a key is released,
    # or something inside the function triggers the end.
        
    # In this case, we don't need to do anything here but wait, since the simulator is its own subprocess.

    def exampleFn():
        time.sleep(0.1)
        return True

    print("Looking at a simulated scene in Sapien.\nPress ESC to exit. (By the way, this is process %s)" % str(os.getpid()))
    
    # Loop the above function until a key is released. For this example, that will be the ESCAPE key.
    
    repeatUntilKey(exampleFn)

    # A clean exit: stop all subprocesses.
    # In general, you can use stopKhafreProcesses to do what it says. Note that by default it will not raise exceptions.
    # If you want to stop processes and handle exceptions yourself, run stopKhafreProcesses(procs, exceptions=True)

    stopKhafreProcesses(procs)

if "__main__" == __name__:
    main()

