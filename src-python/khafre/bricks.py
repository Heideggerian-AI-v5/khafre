import cv2 as cv
import time
from multiprocessing import shared_memory, Process, Lock, Queue
import sys

class _GracefulExit(Exception):
    """
Used to set up a graceful exit on SIGTERM for Khafre subprocesses. Apart
from this role it plays in the file bricks.py, this exception should not
be raised or caught anywhere else.
    """
    pass

def _RequestExit():
    """
Raises a graceful exit exception upon receiving a SIGTERM. Apart from
its role in bricks.py, this function should not appear anywhere else
in your codebase.
    """
    raise _GracefulExit()

class ReifiedProcess():
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
        if isinstance(self._process, Process):
            self._process.kill()
            self.join()
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


