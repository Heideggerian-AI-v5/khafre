import cv2 as cv
from khafre.bricks import ReifiedProcess, RatedSimpleQueue, NameTaken, SHMConsumerPort
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
    def _checkSubscriptionRequest(self, name, queue, consumerSHM):
        if (name in self._subscriptions) or (not isinstance(queue, RatedSimpleQueue)) or (not isinstance(consumerSHM, SHMConsumerPort)):
            return False
        return True
    def _doWork(self):
        """
Loop through the registered input channels. If something
is in a channel, display it (together with frame rate and
dropped frame info).
        """
        for k,v in self._subscriptions.items():
            if (not v.empty()):
                e,rate,dropped = v.getWithRates()
                with v as segImg:
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
    def _cleanup(self):
        """
Loop through registered input channels and flush them.
Also close opened visualization windows.
        """
        def _x(name):
            try:
                cv.destroyWindow(k)
            except:
                pass
        _ = [(_x(k)) for k in self._subscriptions.items()]

