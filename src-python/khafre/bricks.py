import array
import cv2 as cv
import time
from multiprocessing import shared_memory, Process, Lock, Queue, SimpleQueue, Pipe
import signal
import sys

class NameTaken(Exception):
    """
An attempt has been made to associate a name to an entity, when that name
was already associated.
    """
    pass

class _GracefulExit(Exception):
    """
Used to set up a graceful exit on SIGTERM for Khafre subprocesses. Apart
from this role it plays in the file bricks.py, this exception should not
be raised or caught anywhere else.
    """
    pass

def _RequestExit(signum, frame):
    """
Raises a graceful exit exception upon receiving a SIGTERM. Apart from
its role in bricks.py, this function should not appear anywhere else
in your codebase.
    """
    raise _GracefulExit()

class ReifiedProcess:
    """
Wraps together some useful functionality: starting a process, stopping
it on request and do custom cleanup code in such cases.
    """
    def __init__(self):
        self._process=None
        self._daemon=False
    def start(self):
        """
Starts a process associated to this object. Arguments for the process
are member variables of this object.

Note: this function does nothing if a process associated to this object
is already started.

Note: memory is usually not shared between subprocesses. This means that
changing member variables of a ReifiedProcess will have no effect on the
process after it is started. Some subclasses may offer specialized setters
so that the subprocess is informed of the change as well, but unless this
is explicitly stated it should never be assumed.
        """
        if not isinstance(self._process, Process):
            self._process=Process(target=lambda : self._run(), daemon=self._daemon, args=())
            self._process.start()
    def stop(self):
        """
Requests for the process associated with this object to do cleanup and
then terminate. It then joins the process.

Note: it is possible to start another process associated to this object
after termination and joining is complete.
        """
        if isinstance(self._process, Process) and self._process.is_alive():
            #try:
                self._process.terminate()
                self.join()
            #except:
            #    pass
    def join(self, timeout=None):
        """
Joins the process associated to this object. This includes waiting for
it to terminate if it has not done so already.    
        """
        if isinstance(self._process, Process):
            self._process.join(timeout)
        self._process=None
    def doWork(self):
        """
Derived classes should extend this method. This is where the useful
code, i.e. the one that does the actual work, gets put.
        """
        pass
    def cleanup(self):
        """
Derived classes should extend this method. This is where the cleanup
code, i.e. the code to call so that the process can exit gracefully
on request, is put.

Note: it is not required to explicitly terminate daemon subprocesses
started by this process in the cleanup code. This will be done by 
this process' SIGTERM handler (see the except block of _run).
        """
        pass
    def _run(self, *args, **kwargs):
        """
Runs the object's process and sets up graceful exit on SIGTERM.
        """
        signal.signal(signal.SIGTERM, _RequestExit)
        try:
            self.doWork()
        except _GracefulExit:
            self.cleanup()
            # This will take care of terminating all daemon subprocesses started by this process.
            sys.exit(0)

class RatedSimpleQueue:
   """
Wraps around a multiprocessing queue, maintains information about rate of 
incoming and dropped entries. Assumes there is only one consumer of the
queue, and that, when there are several entries in the queue, only the 
latest must be serviced. A "dropped" entry is one that is in the queue 
when another entry shows up.

"Rate of incoming": estimated based on the difference in timestamps
between entries to the queue. Reported in entries/second. Rates above
1 million entries/second are reported as 1 million entries/second.

"Rate of dropped entries": estimated based on dropped frames out of the
last 100. Reported in %.
   """
   def __init__(self):
       self._queue = SimpleQueue()
       self._entriesHistory = array.array('h',[1]*100)
       self._historyHead = 0
       self._previousTS = None
       self._rate=None
   def _markHistory(self,done=True,rewind=False):
       if rewind:
           self._historyHead-=1
           self._historyHead%=100
       m=1
       if not done:
           m=0
       self._entriesHistory[self._historyHead]=m
       self._historyHead+=1
       self._historyHead%=100
   def flush(self):
       while not self._queue.empty():
           self._queue.get()
           self._markHistory()
   def empty(self):
       return self._queue.empty()
   def put(self,e):
       ts = time.perf_counter()
       self._queue.put((ts,e))
   def get(self, block=True, timeout=None):
       if self._queue.empty():
           ts, e = self._queue.get(block=block,timeout=timeout)
           self._markHistory()
       else:
           while not self._queue.empty():
               ts, e = self._queue.get()
               self._markHistory(done=False)
           self._markHistory(done=True,rewind=True)
       if self._previousTS is not None:
           diff = ts - self._previousTS
           if diff>1e-6:
               self._rate = 1.0/(diff)
           else:
               self._rate = 1e6
       self._previousTS = ts
       return e
   def getWithRates(self, block=True, timeout=None):
       e = self.get(block=block, timeout=timeout)
       return e, self._rate, 100-sum(self._entriesHistory)

