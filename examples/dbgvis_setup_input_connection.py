import base64
import cv2 as cv
import keyboard
import numpy
from multiprocessing import shared_memory
import os
from PIL import Image
from websockets.sync.client import connect
import signal
import sys
import time

from khafre.dbgvis import DbgVisualizer

import mss

## DbgVis example: shows how to set up a connection to the khafre debug visualizer.
# In this example, what we will visualize is a resized copy of one of your monitors.

# An auxiliary function to set up a signal handler
goOn={"goOn":True}
def doExit(signum, frame,vp, shm):
    vp.stop()
    shm.close()
    shm.unlink()
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
        # Will want to resize the displayed image.
        #
        # IMPORTANT: the size of the DbgVis window depends on the size of image you send to it!
        #
        imgWidth,imgHeight = (240,int(monitor["height"]*240.0/monitor["width"]))
        # Step 2: set up the shared memory for a connection to the debug visualizer, and initializing a
        # debug visualizer subprocess
        # First, define a shared memory that is "large enough"
        # TODO: write a more elegant way to obtain the size of the necessary buffer
        a = numpy.ones(shape=(imgHeight, imgWidth, 3), dtype=numpy.float32)  # Start with an existing NumPy array
        shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
        # Next, define a numpy array with its buffer located in the shared memory.
        # IMPORTANT: the array must be large enough to hold the image you want it to hold!
        npImage = numpy.ndarray(a.shape, dtype=numpy.float32, buffer=shm.buf)
        # Initialize a DebugVisualizer object.
        vp = DbgVisualizer()
        # Request an input connection to the DbgVisualizer.
        # IMPORTANT: for now, subprocesses in khafre cannot be "hotwired" and you must make all settings, 
        # including all interprocess connection channels, before you start the subprocesses.
        lock, inputChannel = vp.requestInputChannel("Screenshot Cam")
        # Optional, set up signal handlers. As written, the handlers will trigger the termination of the 
        # DbgVisualizer subprocess before the shared memory is closed and unlinked.
        signal.signal(signal.SIGTERM, lambda signum, frame: doExit(signum, frame, vp, shm))
        signal.signal(signal.SIGINT, lambda signum, frame: doExit(signum, frame, vp, shm))
        # Step 3: running the visualizer
        # You can start the debug visualizer process now.
        vp.start()
        # Finally, you can enter a loop in which the screen is captured and sent to the DbgVisualizer.
        # You will also see how many fps your system can manage with this code.
        while True:#goOn["goOn"]:
            resizedScreenshot = getImg(sct, monitor, imgWidth, imgHeight)
            # RECOMMENDATION: write your code so as to minimize the time you keep the input channel lock acquired.
            # IMPORTANT: the lock MUST be released at some point, or it is impossible for DbgVisualizer to
            # claim access to the shared memory!
            with lock:
                numpy.copyto(npImage, resizedScreenshot)
                inputChannel.put(shm.name,imgHeight,imgWidth,3,numpy.float32)
            # Usually, some waiting time between iterations of such a loop would also be needed. However, usually
            # the kind of processes that generate images, such as screenshots and resizes, are "slow", and can
            # function as a delay themselves.
        # Step 4: A clean exit:
        # Stop the debug visualizer first, and only after that close and unlink the shared memory.
        vp.stop()
        shm.close()
        shm.unlink()

if "__main__" == __name__:
    main()

