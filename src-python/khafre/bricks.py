### TODO
# Prepare an alternative set of bricks that use threads instead of processes,
# just to compare ...

import array
import cv2 as cv
import copy
import ctypes
from functools import reduce
import json
import os
from multiprocessing import shared_memory, Condition, Process, Queue, SimpleQueue, Value
import numpy
import time
import signal
import sys
import threading
from threading import Thread
from typing import Optional, Sequence, Union
import weakref

class BadPublisher(Exception):
    """
For whatever reason, a khafre subprocess rejects a publisher request.
    """
    pass

class BadSubscription(Exception):
    """
For whatever reason, a khafre subprocess rejects a subscription request.
    """
    pass

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

def _getProcs(procs, exceptions):
    if isinstance(procs, list) or isinstance(procs, tuple):
        procs = procs
    elif isinstance(procs, dict):
        procs = procs.values()
    else:
        if exceptions:
            raise ValueError("ValueError: procs argument not a list or dictionary")
        procs = []
    return procs

def startKhafreProcesses(procs):
    procs = _getProcs(procs, True)
    _ = [e.start() for e in procs]

def stopKhafreProcesses(procs, exceptions=False):
    """
Utility function to stop khafre reified processes.

By default, function will try to ignore exceptions, i.e. it will return
without doing anything if procs is not a list or dictionary,
and any element of procs that is not a ReifiedProcess is ignored.
Any exceptions thrown by the stopping of a reified process are
also caught and, apart from printing an error message, ignored.

This is an attempt to make it safe to use in signal handlers, with
the intention being that nothing that happens in a signal handler
should raise further exceptions and allow the signal handler to do
its job, e.g. terminate the program.

Input arguments:
    procs: list or dictionary of reified processes to stop.
    """
    def _tryStop(proc, exceptions):
        try:
            proc.stop()
        except Exception as e:
            print("Encountered exception while attempting to stop %s:\n%s" % (str(proc), str(e)))
            if exceptions:
                raise e
    procs = _getProcs(procs, exceptions)
    _ = [_tryStop(e, exceptions) for e in procs if (exceptions or isinstance(e, ReifiedProcess))]

def setSignalHandlers(procs):
    """
Sets up khafre process termination on signal.SIGTERM and signal.SIGINT.

The previously set handlers will be called after stop requests are sent
to khafre reified processes in procs.

Input arguments:
    procs: list or dictionary of ReifiedProcess objects
Output arguments:
    sigintHandler: previous handler for signal.SIGINT
    sigtermHandler: previous handler for signal.SIGTERM
    """
    def _sigHandler(signum, frame, procs, handler):
        stopKhafreProcesses(procs)
        handler(signum, frame)
    sigintHandler = signal.getsignal(signal.SIGINT)
    sigtermHandler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, lambda signum, frame: _sigHandler(signum, frame, procs, sigintHandler))
    signal.signal(signal.SIGTERM, lambda signum, frame: _sigHandler(signum, frame, procs, sigtermHandler))
    return sigintHandler, sigtermHandler

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
       self._queue = Queue()
       self._entriesHistory = array.array('h',[1]*100)
       self._historyHead = 0
       self._previousTS = None
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
       while self._queue.qsize():
           self._queue.get()
           self._markHistory()
   def empty(self):
       return 0 == self._queue.qsize()
   def put(self,e):
       ts = time.perf_counter()
       rate = 0
       if self._previousTS is not None:
           diff = ts - self._previousTS
           if diff>1e-6:
               rate = 1.0/(diff)
           else:
               rate = 1e6
       self._previousTS = ts
       self._queue.put((rate,e))
   def _get(self, block=True, timeout=None):
       if 0 == self._queue.qsize():
           rate, e = self._queue.get()
           self._markHistory()
       else:
           while self._queue.qsize():
               rate, e = self._queue.get()
               self._markHistory(done=False)
           self._markHistory(done=True,rewind=True)
       return rate, e
   def get(self, block=True, timeout=None):
       _, e = self._get(block=block, timeout=timeout)
       return e
   def getWithRates(self, block=True, timeout=None):
       rate, e = self._get(block=block, timeout=timeout)
       return e, rate, 100-sum(self._entriesHistory)

# lambdas and bound methods cannot be pickled on Windows (or perhaps more generally
# on architectures without process forking?)
# Workaround: send the ReifiedProcess as an argument to a function
# that then calls its run function 
def _picklePusher(rp):
    rp._run()

def _closeWire(shm):
    if shm is not None:
        shm.close()
        shm.unlink()

class _Wire:
    def __init__(self, name, shape:Optional[Union[list,tuple]]=None, dtype:Optional[numpy.dtype]=None):
        self._name = name
        self._readerCount = Value(ctypes.c_int32)
        self._state = Value(ctypes.c_int32)
        self._state.value = 0
        self._shm = None
        self._npArray = None
        self._notifications = []
        self._writerEvent = {"event": None}
        self._readerEvents = []
        if (shape is not None) and (dtype is not None):
            buffsize = reduce(lambda x,y: x*y, shape)*dtype(1).nbytes
            self._shm = shared_memory.SharedMemory(create=True, size=buffsize)
            self._npArray = numpy.ndarray(shape, dtype=dtype, buffer=self._shm.buf)
        self._finalizer = weakref.finalize(self,_closeWire,self._shm)
    def isReadyForPublishing(self):
        with self._state:
            aux = self._state.value
        return 0 == aux
    def publish(self, notifData, shmData):
        """
AVOID using this function. Only included for some debug purposes.
        """
        if self._shm is None and (shmData is not None):
            raise ValueError("Attempting to send an array over a wire with no shared memory.")
        if not self.isReadyForPublishing():
            raise AssertionError("Attempting to send before all readers copied previous data.")
        if shmData is not None:
            srcH, srcW = shmData.shape[0], shmData.shape[1]
            if(srcH != self._npArray.shape[0]) or (srcW != self._npArray.shape[1]):
                shmData = cv.resize(shmData, (self._npArray.shape[1], self._npArray.shape[0]), interpolation=cv.INTER_LINEAR)
            numpy.copyto(self._npArray, shmData)
        with self._state: 
            _=[x.put(notifData) for x in self._notifications]
            self._state.value = self._readerCount.value
        for e in self._readerEvents:
            if e is not None:
                with e:
                    e.notify_all()

class _PublisherPort:
    def __init__(self, wire, event):
        self._name = wire._name
        self._shm = None
        self._npArray = None
        self._shmName = None
        self._shape = None
        self._dtype = None
        if wire._npArray is not None:
            self._shmName = wire._shm.name
            self._shape = wire._npArray.shape
            self._dtype = wire._npArray.dtype
        self._state = wire._state
        self._readerCount = wire._readerCount
        self._notifications = wire._notifications
        self._events = wire._readerEvents
        wire._writerEvent["event"] = event
    def hasSHM(self):
        return self._shm is not None
    def isReady(self):
        with self._state:
            aux = self._state.value
        return 0 == aux
    def publish(self, notifData, shmData):
        if self._shmName is None and (shmData is not None):
            raise ValueError("Attempting to send an array over a wire with no shared memory.")
        if not self.isReady():
            raise AssertionError("Attempting to send before all readers copied previous data.")
        if shmData is not None:
            if self._shm is None:
                self._shm = shared_memory.SharedMemory(name=self._shmName)
                self._npArray = numpy.ndarray(self._shape, dtype=self._dtype, buffer=self._shm.buf)
            srcH, srcW = shmData.shape[0], shmData.shape[1]
            if(srcH != self._shape[0]) or (srcW != self._shape[1]):
                shmData = cv.resize(shmData, (self._shape[1], self._shape[0]), interpolation=cv.INTER_LINEAR)
            numpy.copyto(self._npArray, shmData)
        with self._state:
            _=[x.put(notifData) for x in self._notifications]
            self._state.value = self._readerCount.value
        for e in self._events:
            if e is not None:
                with e:
                    e.notify_all()

class _SubscriberPort:
    def __init__(self, wire, event):
        self._name = wire._name
        self._shm = None
        self._npArray = None
        self._shmName = None
        self._shape = None
        self._dtype = None
        if wire._npArray is not None:
            self._shmName = wire._shm.name
            self._shape = wire._npArray.shape
            self._dtype = wire._npArray.dtype
        self._state = wire._state
        self._notification = RatedSimpleQueue()
        self._event = wire._writerEvent
        with wire._readerCount:
            wire._readerCount.value += 1
            wire._notifications.append(self._notification)
            wire._readerEvents.append(event)
    def hasSHM(self):
        return self._shm is not None
    def isReady(self):
        with self._state:
            aux = self._state.value
        return (not self._notification.empty()) and (0 < aux)
    def receive(self):
        if not self.isReady():
            raise AssertionError("Attempting to read before data available.")
        shmData = None
        if self._shmName is not None:
            if self._shm is None:
                self._shm = shared_memory.SharedMemory(name=self._shmName)
                self._npArray = numpy.ndarray(self._shape, dtype=self._dtype, buffer= self._shm.buf)
            shmData = numpy.copy(self._npArray)
        with self._state:
            notifData, fps, dropped = self._notification.getWithRates()
            self._state.value -= 1
        if self._event.get("event") is not None:
            with self._event["event"]:
                self._event["event"].notify_all()
        return notifData, shmData, fps, dropped

class ReifiedProcess:
    """
Wraps together some useful functionality: starting a process, stopping
it on request and do custom cleanup code in such cases.
    """
    def __init__(self):
        self._process=None
        self._daemon=False
        self._command = SimpleQueue()
        self._keepOn=False
        self._subscriptions={}
        self._publishers={}
        self._dataFromSubscriptions={}
        self._dataToPublish={}
        self._event = Condition()
        self._bypassEvent = False
    def havePublisher(self, name):
        return name in self._publishers
    def _setBypassEvent(self):
        self._bypassEvent = True
    def _clearBypassEvent(self):
        self._bypassEvent = False
    def _requestToPublish(self, name, notification, image):
        if name in self._dataToPublish:
            self._dataToPublish[name] = {"ready": True, "notification": notification, "image": image}
    def _requestSubscribedData(self, name):
        return self._dataFromSubscriptions[name].get("notification"), self._dataFromSubscriptions[name].get("image"), self._dataFromSubscriptions[name].get("rate"), self._dataFromSubscriptions[name].get("dropped")
    def sendCommand(self, command, block=False, timeout=None):
        self._command.put(command)#, block=block, timeout=timeout)
        with self._event:
            self._event.notify_all()
    def _handleCommand(self, command):
        """
Subclasses should place command handling code here.

The command queue is checked every iteration. All pending commands
will be run before doWork. Therefore, keep commands easy to handle
OR be about rare events (e.g. loading a model).
        """
        pass
    def _checkPublisherRequest(self, name: str, wire: _Wire):
        """
Subclasses should place here the code that will check whether a request
for a publisher is appropriate (e.g., that a shared memory is provided
when one is expected, that the publisher name is one from a given list,
or that the name was not requested before etc.

Return True if the publisher is acceptable, False otherwise. 
        """
        pass
    def _checkSubscriptionRequest(self, name: str, wire: _Wire):
        """
Subclasses should place here the code that will check whether a request
for a subscription is appropriate (e.g., that a shared memory is provided
when one is expected, that the subscription name is one from a given list,
or that the name was not requested before etc.

Return True if the subscription is acceptable, False otherwise. 
        """
        pass
    def addSubscription(self, name: str, wire: _Wire):
        if self._checkSubscriptionRequest(name, wire):
            self._subscriptions[name] = _SubscriberPort(wire, self._event)
            self._dataFromSubscriptions[name] = {}
        else:
            raise BadSubscription
    def addPublisher(self, name: str, wire: _Wire):
        if self._checkPublisherRequest(name, wire):
            self._publishers[name] = _PublisherPort(wire, self._event)
            self._dataToPublish[name] = {}
        else:
            raise BadPublisher
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
        if (not isinstance(self._process, Process)) or (not self._process.is_active()):
            self._process=Process(target=_picklePusher, daemon=self._daemon, args=(self,))
            self._process.start()
    def stop(self):
        """
Requests for the process associated with this object to do cleanup and
then terminate. It then joins the process.

Note: it is possible to start another process associated to this object
after termination and joining is complete.
        """
        if isinstance(self._process, Process) and self._process.is_alive():
            self._process.terminate()
            self.join()
    def join(self, timeout=None):
        """
Joins the process associated to this object. This includes waiting for
it to terminate if it has not done so already.    
        """
        if isinstance(self._process, Process):
            self._process.join(timeout)
        self._process=None
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
    def _internalEvent(self):
        return False
    def _run(self, *args, **kwargs):
        """
Runs the object's process and sets up graceful exit on SIGTERM.
        """
        signal.signal(signal.SIGTERM, _RequestExit)
        # The main process may be issued a SIGINT from, e.g., the command line via Ctrl+C on Unix
        # systems. It should have a handler that will then initiate subprocess cleanups, and in
        # particular, a handler that sends termination requests to khafre subprocesses before
        # freeing shared memories.
        # However, to avoid that same signal handler from being invoked by the khafre subprocesses
        # themselves, set up an "ignore signal" handler here. Invoking the same signal handler as
        # in the main process will result in error, as a None object will attempt to terminate.
        signal.signal(signal.SIGINT, lambda s,f: None)
        self._keepOn = True
        try:
            self._onStart() # Run process startup code.
            while self._keepOn:
                with self._event: # Check if there is anything to do
                    haveEvent = self._bypassEvent or self._internalEvent() # Maybe process code requests another step immediately
                    haveEvent = haveEvent or all([x.isReady() for x in self._subscriptions.values()]) # Or maybe a full set of inputs is available
                    haveEvent = haveEvent or any([(x.isReady() and (self._dataToPublish[name].get("ready", False))) for name, x in self._publishers.items()]) # Or maybe one of the outputs can be updated
                    haveEvent = haveEvent or (not self._command.empty())
                    if not haveEvent:
                        self._event.wait()
                for name, pub in self._publishers.items():
                    if pub.isReady() and self._dataToPublish[name].get("ready", False):
                        pub.publish(self._dataToPublish[name].get("notification", None), self._dataToPublish[name].get("image", None))
                        self._dataToPublish[name]["ready"] = False
                while not self._command.empty():
                    self._handleCommand(self._command.get())
                fullInput = all([x.isReady() for x in self._subscriptions.values()])
                if self._bypassEvent or fullInput:
                    if True:#fullInput:
                        for name, sub in self._subscriptions.items():
                            if sub.isReady():
                                notification, shmData, fps, dropped = sub.receive()
                                self._dataFromSubscriptions[name] = {"notification": notification, "image": shmData, "rate": fps, "dropped": dropped}
                    self._doWork()
        except _GracefulExit:
            self._cleanup()
            # This will take care of terminating all daemon subprocesses started by this process.
            sys.exit(0)

def drawWire(wireName, publisher, subscribers, shape, dtype, wireList=None):
    '''
Connect various subprocesses via a "wire" -- a shared memory for a numpy array, and notification
queues to indicate when the array should be inspected by subscribers.

The array may be empty, which is signalled by the shape and/or dtype parameters being None. In
this case, the wire only contains notification queues.

Inputs:
    wireName: a string by which the process that created the wire may refer to it.
    publisher: a pair of the form (name, process) or an empty tuple. Name is what the wire is called by
                the publisher process in its internal operation.
    subscribers: a list of pairs of the form (name, process). Name is what the wire is called by
                the subscriber process in its internal operation.
    shape: a tuple indicating a numpy array shape.
    dtype: a numpy data type for the array elements.
    wireList: none or a dictionary with wire names as keys and producer wire objects as values.
Outputs:
    wirelist: an updated wirelist.

IMPORTANT: keep the wire list object for as long as the subprocesses using the wire are running!
The wire object is in the wirelist, and once it is garbage collected, its corresponding shared memory
will be closed and rendered inaccessible to the subprocesses using the wire.

The same process may not appear as both publisher and subscriber. An attempt to do so will be ignored.

The name that a publisher uses for a wire need not be the same as the name used by a subscriber. E.g.,
one may have a publisher sending "Output Image" to the wire while a subscriber calls it "Input Image".
    '''
    def _duplicate(x):
        if x is None:
            return None
        return SHMConsumerPort(x._lock, x._name, x._shape, x._dtype)
    if wireList is None:
        wireList={}
    if wireName in wireList:
        raise NameTaken
    wire = _Wire(wireName, shape=shape, dtype=dtype)
    wireList[wireName] = wire
    pub = None
    if 1 < len(publisher):
        pub = publisher[1]
    for name, s in subscribers:
        if s != pub:
            s.addSubscription(name, wire)
    name, pub = None, None
    if 0 < len(publisher):
        name, pub = publisher
    if 0 < len(publisher):
        pub.addPublisher(name, wire)
    return wireList

class Peeker:
    def __init__(self):
        self._thread=None
        self._subscriptions={}
        self._data={}
        self._lock=threading.Lock()
        self._work=False
        self._event = Condition()
    def addSubscription(self, name, wire):
        self._subscriptions[name] = _SubscriberPort(wire, self._event)
    def _run(self):
        while self._work:
            with self._event:
                haveEvent = any([x.isReady() for x in self._subscriptions.values()])
                if not haveEvent:
                    self._event.wait()
            for name, port in self._subscriptions.items():
                if port.isReady():
                    notification, shmData, fps, dropped = port.receive()
                    self._data[name]={"notification": notification, "image": shmData, "rate": fps, "dropped": dropped}
    def start(self):
        if (self._thread is not None) and (self._thread.is_alive()):
            return
        self._work=True
        self._thread = Thread(target=self._run)
        self._thread.start()
    def stop(self):
        if (self._thread is None) or (not self._thread.is_alive()):
            return
        self._work = False
        self._thread.join()

