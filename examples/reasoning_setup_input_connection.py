import base64
import cv2 as cv
import numpy
import mss
import os
from PIL import Image
import sys
import time

from khafre.bricks import drawWire, setSignalHandlers, startKhafreProcesses, stopKhafreProcesses
from khafre.dbgvis import DbgVisualizer
from khafre.depth import TransformerDepthSegmentationWrapper
from khafre.tracker import ByteTracker
from khafre.contact import ContactDetection
from khafre.optical_flow import OpticalFlow
from khafre.reasoning import Reasoner
from khafre.framestore import YOLOFrameSaver
from khafre.utils import repeatUntilKey

from multiprocessing import Queue

### Object detection/segmentation example: shows how to set up a connection to the khafre object detection,
# and between it and a debug visualizer. See the dbgvis_setup_input_connection.py example for more comments
# on the debug visualizer.
# In this example, we will attempt to have a pretrained YOLO model recognize objects visible on screen.

# Define a function to capture the image from the screen.
def getImg(sct, monitor):
    sct_img = sct.grab(monitor)
    pilImage = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    rawImage = numpy.ascontiguousarray(numpy.float32(numpy.array(pilImage)[:,:,(2,1,0)]))/255.0
    return rawImage

def main():

    percIntTheory = "theories/example_cup_on_table/perceptionInterpretation.dfl"
    updSchTheory = "theories/example_cup_on_table/updateSchemas.dfl"
    connQTheory = "theories/example_cup_on_table/connectivityQueries.dfl"
    schClsTheory = "theories/example_cup_on_table/schemaClosure.dfl"
    schIntTheory = "theories/example_cup_on_table/schemaInterpretation.dfl"
    updQTheory = "theories/example_cup_on_table/updateQueries.dfl"
    interpMasksTheory = "theories/example_cup_on_table/interpretMasks.dfl"
    storeMasksTheory = "theories/example_cup_on_table/storeMasks.dfl"
    backgroundFacts = "theories/example_cup_on_table/backgroundFacts.dfl"
    defaultFacts = "theories/example_cup_on_table/defaultFacts.dfl"

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
        procs["objP"] = ByteTracker()
        # Monocular depth estimation is VERY computationally expensive, try to have it run on the GPU
        procs["dptP"] = TransformerDepthSegmentationWrapper(device="cuda")
        procs["conP"] = ContactDetection()
        procs["optP"] = OpticalFlow()
        procs["reasoner"] = Reasoner()
        procs["storage"] = YOLOFrameSaver()

        # Set up the connections to dbg visualizer and object detection.

        drawWire("Screenshot Cam", (), [("Screenshot Cam", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Input Image", (), [("InpImg", procs["objP"]), ("InpImg", procs["dptP"]), ("InpImg", procs["optP"]), ("InpImg", procs["reasoner"])], (imgHeight, imgWidth, 3), numpy.uint8, wireList=wireList)
        drawWire("Dbg Obj Seg", ("DbgImg", procs["objP"]), [("Object Detection/Segmentation", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Dbg Depth", ("DbgImg", procs["dptP"]), [("Depth Estimation", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Mask Image", ("OutImg", procs["objP"]), [("MaskImg", procs["conP"]), ("MaskImg", procs["optP"]), ("MaskImg", procs["reasoner"])], (imgHeight, imgWidth), numpy.uint16, wireList=wireList)
        drawWire("Depth Image", ("OutImg", procs["dptP"]), [("DepthImg", procs["conP"]), ("DepthImg", procs["optP"])], (imgHeight, imgWidth), numpy.float32, wireList=wireList)
        drawWire("Dbg Contact", ("DbgImg", procs["conP"]), [("Contact Detection", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("Dbg Optical Flow", ("DbgImg", procs["optP"]), [("Optical Flow (sparse)", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        drawWire("OutImg Contact", ("OutImg", procs["conP"]), [("ContactMask", procs["reasoner"])], (imgHeight, imgWidth), numpy.uint32, wireList=wireList)
        drawWire("OutImg OpticalFlow", ("OutImg", procs["optP"]), [("Optical Flow (sparse)", procs["reasoner"])], (imgHeight, imgWidth), numpy.uint32, wireList=wireList)
        drawWire("Masks to Store", ("OutImg", procs["reasoner"]), [("InpImg", procs["storage"])], (imgHeight, imgWidth, 3), numpy.uint8, wireList=wireList)
        drawWire("Dbg Storage", ("DbgImg", procs["storage"]), [("Frame Storage", procs["dbgP"])], (imgHeight, imgWidth, 3), numpy.float32, wireList=wireList)
        
        procs["reasoner"].registerWorker("contact", procs["conP"])
        procs["reasoner"].registerWorker("opticalFlow", procs["optP"])

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

        # Define and run some code that actually does something with the set up processes.

        # This function will be called repeatedly until some condition happens: either a key is released,
        # or something inside the function triggers the end.
        
        # This function grabs the screen and sends it to the various processes.
        # You will also see how many fps your system can manage with this code.

        def exampleFn(sct, monitor, wireList):
            
            screenshot = getImg(sct, monitor)

            # Send the screenshot to dbgvis and object detection.
            if wireList["Screenshot Cam"].isReadyForPublishing():
                wireList["Screenshot Cam"].publish("Hello World!", screenshot)
            if wireList["Input Image"].isReadyForPublishing():
                wireList["Input Image"].publish({"imgId": str(time.perf_counter())}, (screenshot*255).astype(numpy.uint8))
            return True
                
        print("Starting object segmentation and depth estimation will take a while, wait a few seconds for debug windows for them to show up.\nPress ESC to exit. (By the way, this is process %s)" % str(os.getpid()))

        # Loop the above function until a key is released. For this example, that will be the ESCAPE key.
        
        repeatUntilKey(lambda : exampleFn(sct, monitor, wireList))

        # A clean exit: stop all subprocesses.
        # In general, you can use stopKhafreProcesses to do what it says. Note that by default it will not raise exceptions.
        # If you want to stop processes and handle exceptions yourself, run stopKhafreProcesses(procs, exceptions=True)

        stopKhafreProcesses(procs)

if "__main__" == __name__:
    main()

