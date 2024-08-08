import base64
import cv2 as cv
import numpy
import mss
import os
from PIL import Image
from pynput.keyboard import Key, Listener
import signal
import sys
import time

from khafre.bricks import SHMPort, RatedSimpleQueue, drawWire, setSignalHandlers, startKhafreProcesses, stopKhafreProcesses
from khafre.dbgvis import DbgVisualizer
from khafre.utils import repeatUntilKey


## DbgVis example: shows how to set up a connection to the khafre debug visualizer.
# In this example, what we will visualize is a resized copy of one of your monitors.
# The main process -- this one -- will capture the screen, rescale it, and send it
# to a subprocess: the khafre debug visualizer.
# The CPU usage of this process will be 100%: it will send images as fast as it can.
# The debug visualizer meanwhile doesn't need to do much and its CPU usage will be
# much lower.

# Define a function to capture the image from the screen.
def getImg(sct, monitor):
    sct_img = sct.grab(monitor)
    pilImage = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    rawImage = numpy.ascontiguousarray(numpy.float32(numpy.array(pilImage)[:,:,(2,1,0)]))/255.0
    return rawImage

def main():

    # IMPORTANT: this will be our registry of "wires", connections between khafre subprocesses.
    # Keep this variable alive at least as long as the subprocesses are running.

    wireList = {}
    
    # For ease of start up, cleanup, and setting up termination handlers, process objects should be stored in a dictionary.
    
    procs = {}

    # Step 1: setting up your data source
    # In this case, the data source is one of your monitors, so will set up some variables 
    # to allow screen capture.

    with mss.mss() as sct:

        # Get data about the "first" monitor. Note: index 0 is the "All in one" monitor.

        monitor = sct.monitors[1]

        # We will want to downsize the displayed image.
        #
        # IMPORTANT: the size of the DbgVis window depends on the size of image you send to it!
        #

        imgWidth,imgHeight = (240,int(monitor["height"]*240.0/monitor["width"]))
        
        # Step 2: initialize a DebugVisualizer object.
        procs["vp"] = DbgVisualizer()

        drawWire("Screenshot Cam", [], [("Screenshot Cam", procs["vp"])], (imgHeight, imgWidth, 3), numpy.float32, RatedSimpleQueue, wireList=wireList)

        # Optional, but STRONGLY recommended: set up signal handlers. The handlers will trigger the 
        # termination of the DbgVisualizer subprocess. Alternatively, ensure in some other way that
        # subprocesses are terminated at exit.
        # Note that the previously registered sigint and sigterm handlers are returned, so you can
        # restore them if you need to. That may be the case when you stop the khafre processes manually,
        # and then wish to continue running your program anyway.

        sigintHandler, sigtermHandler = setSignalHandlers(procs)
        
        # Step 3: running the visualizer
        # You can start the debug visualizer process now.

        startKhafreProcesses(procs)

        # Step 4: define and run some code that actually does something with the set up processes.

        # This function will be called repeatedly until some condition happens: either a key is released,
        # or something inside the function triggers the end.
        
        # This function grabs the screen and sends it to the debug visualizer.
        # You will also see how many fps your system can manage with this code.

        def exampleFn(sct, monitor, wireList):

            screenshot = getImg(sct, monitor)

            # Place the image into the shared port. Also, notify the consumer (DbgVis) that something happened.
            # Note: screenshot is likely larger than the producer's image. producer.send will automatically resize
            # in this case.

            # DbgVis can also print something for us. We could also have sent a notification with an empty string
            # instead.
            wireList["Screenshot Cam"].publish(screenshot, "Hello World!")
                
            # Usually, some waiting time between iterations of such a loop would also be needed. However, usually
            # the kind of processes that generate images, such as screenshots and resizes, are "slow", and can
            # function as a delay themselves.
            
            # Returning False will also exit the loop. For this example, we don't need this.
            return True
        
        print("Press ESC to exit. (By the way, this is process %s)" % str(os.getpid()))

        # Loop the above function until a key is released. For this example, that will be the ESCAPE key.
        
        repeatUntilKey(lambda : exampleFn(sct, monitor, wireList))

        # Step 5: A clean exit:
        # Stop and join the debug visualizer.
        # In general, you can use stopKhafreProcesses to do what it says. Note that by default it will not raise exceptions.
        # If you want to stop processes and handle exceptions yourself, run stopKhafreProcesses(procs, exceptions=True)

        stopKhafreProcesses(procs)

if "__main__" == __name__:
    main()

