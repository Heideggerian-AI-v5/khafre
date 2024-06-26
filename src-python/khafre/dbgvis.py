import cv2 as cv
from khafre.bricks import ReifiedProcess, RatedSimpleQueue, NameTaken
from multiprocessing import Lock, shared_memory
import numpy

class DbgVisualizer(ReifiedProcess):
    """
Implements a process that will display images in OpenCV
windows. These images are passed to the process as numpy
arrays in shared memories. Information that a new image
is to be displayed is given in queues ("input channels").

Note that this process also writes to the shared memory
(to display frame rate and dropped frames info). Thus,
users of the process should only use buffers associated
to DbgVisualizer input channels for debugging-related 
images, not images used in other perception subprocesses.
    """
    def __init__(self):
        super().__init__()
        self._inputs = {}
    def requestInputChannel(self, name, consumerSHMPort):
        """
Creates a Lock and RatedSimpleQueue and associates them to name,
which will also be used as the title of the OpenCV window.
No other input channel must use the same name, and if name
is already used then the function fails.

A user of this input channel must employ Lock to synchronize
a shared memory.
        """
        if name in self._inputs:
            raise NameTaken
        self._inputs[name] = (consumerSHMPort, RatedSimpleQueue())
        return self._inputs[name][1]
    def doWork(self):
        """
Loop through the registered input channels. If something
is in a channel, display it (together with frame rate and
dropped frame info).
        """
        for k,v in self._inputs.items():
            consumer, rsq = v
            if not rsq.empty():
                e,rate,dropped = rsq.getWithRates()
                with consumer as segImg:
                    rateAdj = rate
                    if rate is None:
                        rateAdj = 0.0
                    rateStr = "%.02f fps | %d%% dbg drop" % (rateAdj, dropped)
                    (text_width, text_height), baseline = cv.getTextSize(rateStr, cv.FONT_HERSHEY_SIMPLEX, 0.5, cv.LINE_AA)
                    cv.rectangle(segImg,(0,0),(text_width, text_height+baseline),(0.0,0.0,0.0),-1)
                    cv.putText(segImg,rateStr,(0, text_height), cv.FONT_HERSHEY_SIMPLEX, 0.5, (1.0,1.0,0.0), 1, cv.LINE_AA)
                    e = str(e)
                    if "" != e:
                        (e_width, e_height), e_baseline = cv.getTextSize(e, cv.FONT_HERSHEY_SIMPLEX, 0.5, cv.LINE_AA)
                        cv.rectangle(segImg,(0,text_height+baseline),(e_width, text_height+baseline+e_height),(0.0,0.0,0.0),-1)
                        cv.putText(segImg,e,(0,text_height+e_height), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0.0,0.5,1.0), 1, cv.LINE_AA)
                    cv.imshow(k, segImg)
        cv.waitKey(10)
    def cleanup(self):
        """
Loop through registered input channels and flush them.
Also close opened visualization windows.
        """
        def _x(name):
            try:
                cv.destroyWindow(k)
            except:
                pass
        _ = [(_x(k), v[1].flush()) for k,v in self._inputs.items()]

