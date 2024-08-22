import cv2 as cv
from khafre.bricks import ReifiedProcess
from multiprocessing import Queue
import numpy

class ImageSource(ReifiedProcess):
    def __init__(self):
        super().__init__()
        self._dbgImg = None
        self._bypassEvent = True
        self._outputNames = {"OutImg", "DbgImg"}
    def _checkPublisherRequest(self, name, wire):
        return name in self._outputNames
    def _checkSubscriptionRequest(self, name, wire):
        return False
    def _onStart(self):
        """
Derived classes should extend this method. This is where startup code
should be placed.

In particular, code that constructs objects so as to acquire resources
should be in here. This is a somewhat fuzzy notion which will require
developper judgement, but some examples will help clarify this. It
also helps understanding to consider that when a subprocess starts, it
will have access to copies of the objects available at its creation
(including this object). Therefore, to decide whether something should
be done in __init__ or here, ask yourself:

1) can this thing be copied at all? Not all python objects can be copied.
In particular, if something cannot be pickled, it cannot be copied for
access by a subprocess.
2) is it wasteful to copy this thing? Some objects can be copied, but will
be so large that one wants to avoid having useless copies of them. Remember,
the main process copy of this reified object will mostly be sitting idle!
Also, some objects imply the use of some system resource (e.g. file handles)
that may be limited. Avoid encumbering the system too much.

Some examples:
1) numeric paremeters: safe to have them in __init__.
2) helper thread object: might be declared in __init__, but definitely only
start it in onStart!
3) neural network model: these things tend to be huge, so better in onStart.
Even better, have models loadable during the handling of some command.
        """
        pass
    def _doWork(self):
        """
Derived classes should extend this method. This is where the useful
code, i.e. the one that does the actual work, gets put.

IMPORTANT: this function will be enclosed in a loop that runs (almost)
as long as the process does. This function MUST terminate, e.g. it
should not include any while True loops.
        """
        pass
    def _cleanup(self):
        """
Derived classes should extend this method. This is where the cleanup
code, i.e. the code to call so that the process can exit gracefully
on request, is put.

Note: it is not required to explicitly terminate daemon subprocesses
started by this process in the cleanup code. This will be done by 
this process' SIGTERM handler (see the except block of _run).
        """
        pass
