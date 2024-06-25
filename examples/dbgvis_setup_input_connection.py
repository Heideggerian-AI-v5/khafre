import base64
import cv2 as cv
import numpy
import mss
from multiprocessing import shared_memory
import os
from PIL import Image
from pynput.keyboard import Key, Listener
import signal
import sys
import time

from khafre.bricks import SHMPort
from khafre.dbgvis import DbgVisualizer


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
def doExit(signum, frame, vp):
    vp.stop()
    sys.exit()

# Define a function to capture the image from the screen and resize it.
def getImg(sct, monitor, imgWidth, imgHeight):
    sct_img = sct.grab(monitor)
    pilImage = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    rawImage = numpy.ascontiguousarray(numpy.float32(numpy.array(pilImage)[:,:,(2,1,0)]))/255.0
    return cv.resize(rawImage, (imgWidth, imgHeight), interpolation=cv.INTER_LINEAR)

def main():

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
        
        # Step 2: set up the shared memory for a connection to the debug visualizer, and initializing a
        # debug visualizer subprocess
        # First, define a shared memory port for numpy arrays of appropriate size.
        # Note, both producer and consumer(s) can write to the shared array. The producer is the one responsible
        # for freeing it. 

        producer, consumer = SHMPort((imgHeight, imgWidth, 3), numpy.float32)

        # Initialize a DebugVisualizer object.
        vp = DbgVisualizer()

        # Request an input connection to the DbgVisualizer. This is used to notify it that something
        # changed on the shared memory port and needs attention.
        # IMPORTANT: subprocesses in khafre cannot be "hotwired" and you must make all settings, 
        # including all interprocess connection channels, before you start the subprocesses.

        inputChannel = vp.requestInputChannel("Screenshot Cam", consumer)

        # Optional, but STRONGLY recommended: set up signal handlers. The handlers must trigger the 
        # termination of the DbgVisualizer subprocess. Alternatively, ensure in some other way that
        # subprocesses are terminated at exit.

        signal.signal(signal.SIGTERM, lambda signum, frame: doExit(signum, frame, vp))
        signal.signal(signal.SIGINT, lambda signum, frame: doExit(signum, frame, vp))

        # Step 3: running the visualizer
        # You can start the debug visualizer process now.

        vp.start()

        # Finally, you can enter a loop in which the screen is captured and sent to the DbgVisualizer.
        # You will also see how many fps your system can manage with this code.
        # Just as an example, setting up a way to exit the loop without signals as well -- in this case,
        # when releasing the ESC key.

        print("Press ESC to exit. (By the way, this is process %s)" % str(os.getpid()))
        with Listener(on_press=on_press, on_release=on_release) as listener:
            while goOn["goOn"]:
                resizedScreenshot = getImg(sct, monitor, imgWidth, imgHeight)

                # Place the image into the shared port. Also, notify the consumer (DbgVis) that something happened.

                producer.send(resizedScreenshot)
                inputChannel.put(True)

                # Usually, some waiting time between iterations of such a loop would also be needed. However, usually
                # the kind of processes that generate images, such as screenshots and resizes, are "slow", and can
                # function as a delay themselves.

            listener.join()

        # Step 4: A clean exit:
        # Stop the debug visualizer.

        vp.stop()

if "__main__" == __name__:
    main()

