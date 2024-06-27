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

from khafre.bricks import RatedSimpleQueue, SHMPort
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

    with mss.mss() as sct:

        monitor = sct.monitors[1]

        imgWidth,imgHeight = (int(monitor["width"]*240.0/monitor["height"]), 240)
        
        # We will need a debug visualizer and object detection processes.
        # We need to send a screenshot to the object detection, object detection will send a debug image
        # to the visualizer, and we will also send the screenshot from here to the debug visualizer.
        # Usually, the output from object detection will go somewhere else too, but for this example we will ignore it.
        # We therefore set up the following shared memories:
        
        rawImgProducerPort, rawImgConsumerPort = SHMPort((imgHeight, imgWidth, 3), numpy.uint8) # input image for object detection
        screenshotProducerPort, screenshotConsumerPort = SHMPort((imgHeight, imgWidth, 3), numpy.float32) # screenshot to show in dbgvis
        dbgImgProducerPort, dbgImgConsumerPort = SHMPort((imgHeight, imgWidth, 3), numpy.float32) # segmentation mask image to show in dbgvis
        
        # Writing to shared memories will not notify their users. We need another way to send notifications:
        inputNotification = RatedSimpleQueue() # Notification from this process to object detection: "a screenshot is ready"
        outputQueue = Queue() # In order for object detection to do anything, it must have somewhere to send output

        # Construct process objects. These are not started yet.
        
        dbgP = DbgVisualizer()
        objP = YOLOObjectSegmentationWrapper()

        # Set up DbgVis so that it will use the shared memories where we send images to it. It will provide us with notification queues
        # to inform it when an image is ready.
        
        dbgNotificationQueue = dbgP.requestInputChannel("Object Detection/Segmentation", dbgImgConsumerPort)
        dbgScreenshotNotificationQueue = dbgP.requestInputChannel("Screen capture", screenshotConsumerPort)

        # Set up the connections to object detection.
        
        objP.setInputImagePort(rawImgConsumerPort, inputNotification)
        objP.setOutputImagePort(None, outputQueue)
        objP.setOutputDbgImagePort(dbgImgConsumerPort, dbgNotificationQueue)

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

                # Ignore output from object detection.
                            
                while not outputQueue.empty():
                    outputQueue.get()
                
                screenshot = getImg(sct, monitor)

                # Send the screenshot to dbgvis and object detection.
                
                rawImgProducerPort.send((screenshot*255).astype(numpy.uint8))
                inputNotification.put(True)
                screenshotProducerPort.send(screenshot)
                dbgScreenshotNotificationQueue.put("")
                
            listener.join()

        # A clean exit: stop all subprocesses.
        
        objP.stop()
        dbgP.stop()

if "__main__" == __name__:
    main()

