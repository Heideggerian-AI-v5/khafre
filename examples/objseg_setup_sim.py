import base64
import cv2 as cv
import numpy
from multiprocessing import Queue
import os
from PIL import Image
from pynput.keyboard import Key, Listener
import signal
import sys
import time

from khafre.bricks import RatedSimpleQueue, SHMPort, drawWire
from khafre.dbgvis import DbgVisualizer
from khafre.depth import TransformerDepthSegmentationWrapper
from khafre.segmentation import YOLOObjectSegmentationWrapper
from khafre.contact import ContactDetection
from khafre.optical_flow import OpticalFlow
from khafre.sapien_wrapper import SapienSim

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
def doExit(signum, frame, sim, dbgP, objP):
    sim.stop()
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

    # TODO: have this written in some other way, e.g. by having this relative to this file
    asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../assets")

    wireList={}

    imgWidth,imgHeight = 640,480

    sim = SapienSim(height=imgHeight,width=imgWidth)
    dbgP = DbgVisualizer()
    objP = YOLOObjectSegmentationWrapper()

    # Set up the connections to dbg visualizer and object detection.

    drawWire("Simulation Cam", [("DbgImg", sim)], [("Simulation Cam", dbgP)], (imgHeight, imgWidth, 3), numpy.float32, RatedSimpleQueue, wireList=wireList)
    drawWire("Input Image", [("OutImg", sim)], [("InpImg", objP)], (imgHeight, imgWidth, 3), numpy.uint8, RatedSimpleQueue, wireList=wireList)
    drawWire("Dbg Obj Seg", [("DbgImg", objP)], [("Object Detection/Segmentation", dbgP)], (imgHeight, imgWidth, 3), numpy.float32, RatedSimpleQueue, wireList=wireList)

    # Optional, but STRONGLY recommended: set signal handlers that will ensure the subprocesses terminate on exit.
    signal.signal(signal.SIGTERM, lambda signum, frame: doExit(signum, frame, sim, dbgP, objP))
    signal.signal(signal.SIGINT, lambda signum, frame: doExit(signum, frame, sim, dbgP, objP))
    
    # Only after all connection objects -- shared memories and notification queues -- are set up, we can start.
    dbgP.start()
    objP.start()
    sim.start()
    sim.sendCommand(["SET AMBIENT LIGHT",[[0.5, 0.5, 0.5]]])
    sim.sendCommand(["ADD DIRECTIONAL LIGHT",[[0, 1, -1], [0.5, 0.5, 0.5]]])
    sim.sendCommand(["LOAD ASSET", ["table", asset_path+"table/table.urdf", [0, 0, 0.44], [1, 0, 0, 0]])
    sim.sendCommand(["LOAD ACTOR", ["mug", asset_path+"beermug/BeerMugCollision.obj", asset_path+"beermug/BeerMugVisual.obj", [-0.2, 0, 0.44 + 0.05], [1, 0, 0, 0]])
    sim.sendCommand(["SET CAMERA POSE", [np.array([-2, -2, 3])]])

    with Listener(on_press=on_press, on_release=on_release) as listener:
            while goOn["goOn"]:
                time.sleep(0.01)
                pass
            listener.join()
    sim.stop()
    dbgP.stop()
    objP.stop()

if "__main__" == __name__:
    main()

