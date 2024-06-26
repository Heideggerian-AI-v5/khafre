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
        
        rawImgProducerPort, rawImgConsumerPort = SHMPort((imgHeight, imgWidth, 3), numpy.float32)
        screenshotProducerPort, screenshotConsumerPort = SHMPort((imgHeight, imgWidth, 3), numpy.float32)
        dbgImgProducerPort, dbgImgConsumerPort = SHMPort((imgHeight, imgWidth, 3), numpy.float32)
        inputNotification = RatedSimpleQueue()
        outputQueue = Queue()

        dbgP = DbgVisualizer()
        objP = YOLOObjectSegmentationWrapper()

        dbgNotificationQueue = dbgP.requestInputChannel("Object Detection/Segmentation", dbgImgConsumerPort)
        dbgScreenshotNotificationQueue = dbgP.requestInputChannel("Screen capture", screenshotConsumerPort)

        objP.setInputImagePort(rawImgConsumerPort, inputNotification)
        objP.setOutputImagePort(None, outputQueue)
        objP.setOutputDbgImagePort(dbgImgConsumerPort, dbgNotificationQueue)


        signal.signal(signal.SIGTERM, lambda signum, frame: doExit(signum, frame, dbgP, objP))
        signal.signal(signal.SIGINT, lambda signum, frame: doExit(signum, frame, dbgP, objP))

        dbgP.start()
        objP.start()
        
        objP.sendCommand(("LOAD", ("yolov8n-seg.pt",)))

        print("Starting object segmentation will take a while, wait a few seconds for a debug window labeled \"Object Detection/Segmentation\".\nPress ESC to exit. (By the way, this is process %s)" % str(os.getpid()))
        with Listener(on_press=on_press, on_release=on_release) as listener:
            while goOn["goOn"]:
            
                while not outputQueue.empty():
                    outputQueue.get()
                
                screenshot = getImg(sct, monitor)

                rawImgProducerPort.send(screenshot)
                inputNotification.put(True)
                screenshotProducerPort.send(screenshot)
                dbgScreenshotNotificationQueue.put("")
                
            listener.join()

        objP.stop()
        dbgP.stop()

if "__main__" == __name__:
    main()

