### TODO
# Prepare an alternative set of bricks that use threads instead of processes,
# just to compare ...

import array
import cv2 as cv
import copy
from functools import reduce
import os
from multiprocessing import shared_memory, Process, RLock, Queue, SimpleQueue, Pipe
import numpy
import time
import signal
import sys
import threading
from threading import Thread
from typing import Sequence, Union
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
       if self._queue.empty():
           rate, e = self._queue.get(block=block,timeout=timeout)
           self._markHistory()
       else:
           while not self._queue.empty():
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
# Workaround: send the reifiedObject as an argument to a function
# that then calls its run function 
def picklePusher(rp):
    rp._run()

def _closeSHMProducerPort(shm,lock,active):
    shm.close()
    shm.unlink()
    active["active"] = False
    # We may have released the lock already, in which case just ignore the exception.
    try:
        lock.release()
    except:
        pass
    
class SHMProducerPort:
    """
Defines a "producer port" for sharing a numpy array between subprocesses.
The numpy array size and type must be known at construction and may not
be changed afterwards. The array element type must be a numeric type.

The producer is responsible for allocating and freeing up the shared memory.

The consumers may write in the shared memory as well.

Note, consumers are NOT automatically notified about updates to the shared memory.
Some other mechanism, e.g. a queue or pipe, is needed.

It is possible to either copy a numpy array wholesale via the send function,
or to access the numpy array inside a with block.

A typical pattern is to create a pair of producer and consumer ports with the
SHMPort() function. The consumer port object may be used to initialize several
subprocesses. In this case, the size of the shared buffer will never change.

Producers must be created by a "main" process -- or in any case, if a process
creates a producer, then it must not give it to a khafre subprocess. Khafre
subprocesses must use consumer ports instead.

It is possible to implement a producer/consumer relationship that allows
consumers to be fed variable-sized buffers. It is however more difficult to
write and its cost/benefit ratio may not be favorable in most cases.
    """
    def __init__(self, shape:Union[list,tuple], dtype:numpy.dtype):
        """
Initialize the SHMProducerPort object.

    shape: tuple, a numpy array shape
    dtype: numpy.dtype, a numeric numpy data type for the array elements
        """
        buffsize = reduce(lambda x,y: x*y, shape)*dtype(1).nbytes
        self._shm = shared_memory.SharedMemory(create=True, size=buffsize)
        self._npArray = numpy.ndarray(shape, dtype=dtype, buffer=self._shm.buf)
        self._lock = RLock()
        self._active={"active":True}
        self._finalizer = weakref.finalize(self,_closeSHMProducerPort,self._shm,self._lock,self._active)
    def send(self, src):
        """
Write the contents of array src to the shared memory buffer.

Note, src must have the shame shape and data type as used to create this port. No checks are
performed however.
        """
        if not self._active["active"]:
            raise ValueError
        srcH, srcW = src.shape[0], src.shape[1]
        if(srcH != self._npArray.shape[0]) or (srcW != self._npArray.shape[1]):
            src = cv.resize(src, (self._npArray.shape[1], self._npArray.shape[0]), interpolation=cv.INTER_LINEAR)
        with self._lock:
            numpy.copyto(self._npArray, src)
    def __enter__(self):
        if not self._active["active"]:
            raise ValueError
        self._lock.acquire()
        return self._npArray
    def __exit__(self, exit_type, value, traceback):
        self._lock.release()
        return False

class SHMConsumerPort:
    """
Defines a "consumer port" for sharing a numpy array between subprocesses.

A consumer may write to the shared region, or copy a numpy array wholesale via
the send method. The difference to the producer is that a consumer is not responsible 
for deallocating the shared memory.

Typically, a consumer port is created paired with a producer port, via the
SHMPort() function.
    
    """
    def __init__(self, lock, name, shape, dtype):
        self._lock = lock
        self._shm = None
        self._name = name
        self._shape = shape
        self._dtype = dtype
    def send(self, src):
        srcH, srcW = src.shape[0], src.shape[1]
        self._shm = shared_memory.SharedMemory(name=self._name)
        npArray = numpy.ndarray(self._shape, dtype=self._dtype, buffer=self._shm.buf)
        if(srcH != self._shape[0]) or (srcW != self._shape[1]):
            src = cv.resize(src, (self._shape[1], self._shape[0]), interpolation=cv.INTER_LINEAR)
        with self._lock:
            numpy.copyto(npArray, src)
    def __enter__(self):
        self._lock.acquire()
        self._shm = shared_memory.SharedMemory(name=self._name)
        return numpy.ndarray(self._shape, dtype=self._dtype, buffer=self._shm.buf)
    def __exit__(self, exit_type, value, traceback):
        self._shm.close()
        self._lock.release()
        return False

def SHMPort(shape:Union[list,tuple], dtype:numpy.dtype):
    producer = SHMProducerPort(shape,dtype)
    consumer = SHMConsumerPort(producer._lock, producer._shm.name, shape, dtype)
    return producer, consumer

class ReifiedProcess:
    """
Wraps together some useful functionality: starting a process, stopping
it on request and do custom cleanup code in such cases.
    """
    def __init__(self):
        self._process=None
        self._daemon=False
        self._keepOn=False
        self._subscriptions={}
        self._publishers={}
    def _checkPublisherRequest(self, name, queues, consumerSHM):
        """
Subclasses should place here the code that will check whether a request
for a publisher is appropriate (e.g., that a shared memory is provided
when one is expected, that the publisher name is one from a given list,
or that the name was not requested before etc.

Return True if the publisher is acceptable, False otherwise. 
        """
        pass
    def _checkSubscriptionRequest(self, name, queue, consumerSHM):
        """
Subclasses should place here the code that will check whether a request
for a subscription is appropriate (e.g., that a shared memory is provided
when one is expected, that the subscription name is one from a given list,
or that the name was not requested before etc.

Return True if the subscription is acceptable, False otherwise. 
        """
        pass
    def addSubscription(self, name: str, queue: Union[Queue, SimpleQueue, RatedSimpleQueue], consumerSHM: Union[SHMConsumerPort, None]):
        if self._checkSubscriptionRequest(name, queue, consumerSHM):
            self._subscriptions[name] = SubscriberConsumerWire(consumerSHM, queue)
        else:
            raise BadSubscription
    def addPublisher(self, name: str, queues, consumerSHM: Union[SHMConsumerPort, None]):
        if self._checkPublisherRequest(name, queues, consumerSHM):
            self._publishers[name] = PublisherConsumerWire(consumerSHM, queues)
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
            self._process=Process(target=picklePusher, daemon=self._daemon, args=(self,))
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
    def doWork(self):
        """
Derived classes should extend this method. This is where the useful
code, i.e. the one that does the actual work, gets put.

IMPORTANT: this function will be enclosed in a loop that runs (almost)
as long as the process does. This function MUST terminate, e.g. it
should not include any while True loops.
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
            while self._keepOn:
                self.doWork()
        except _GracefulExit:
            self.cleanup()
            # This will take care of terminating all daemon subprocesses started by this process.
            sys.exit(0)

class ProducerWire():
    def __init__(self, producer, notifications):
        self._producer = producer
        self._notifications = notifications
    def hasSHM(self):
        return self._producer is not None
    def sendNotification(self, data):
        _=[x.put(data) for x in self._notifications]
    def publish(self, shmData, notifData):
        if (self._producer is not None) and (shmData is not None):
            self._producer.send(shmData)
        self.sendNotifications(notifData)
    def sendNotifications(self, data):
        _=[x.put(data) for x in self._notifications]
    def send(self, data):
        self._producer.send(data)
    def __enter__(self):
        return self._producer.__enter__()
    def __exit__(self, exit_type, value, traceback):
        return self._producer.__exit__(exit_type, value, traceback)

class PublisherConsumerWire():
    def __init__(self, consumer, notifications):
        self._consumer = consumer
        self._notifications = notifications
    def hasSHM(self):
        return self._consumer is not None
    def sendNotification(self, data):
        _=[x.put(data) for x in self._notifications]
    def publish(self, shmData, notifData):
        if (self._consumer is not None) and (shmData is not None):
            self._consumer.send(shmData)
        self.sendNotifications(notifData)
    def sendNotifications(self, data):
        _=[x.put(data) for x in self._notifications]
    def send(self, data):
        self._consumer.send(data)
    def __enter__(self):
        return self._consumer.__enter__()
    def __exit__(self, exit_type, value, traceback):
        return self._consumer.__exit__(exit_type, value, traceback)

class SubscriberConsumerWire():
    def __init__(self, consumer, notification):
        self._consumer = consumer
        self._notification = notification
    def hasSHM(self):
        return self._consumer is not None
    def empty(self):
        return self._notification.empty()
    def get(self):
        return self._notification.get()
    def getWithRates(self):
        return self._notification.getWithRates()
    def __enter__(self):
        return self._consumer.__enter__()
    def __exit__(self, exit_type, value, traceback):
        return self._consumer.__exit__(exit_type, value, traceback)

def drawWire(wireName, publishers, subscribers, shape, dtype, qtype, wireList=None):
    '''
Connect various subprocesses via a "wire" -- a shared memory for a numpy array, and notification
queues to indicate when the array should be inspected by subscribers.

The array may be empty, which is signalled by the shape and/or dtype parameters being None. In
this case, the wire only contains notification queues.

Inputs:
    wireName: a string by which the process that created the wire may refer to it.
    publishers: a list of pairs of the form (name, process). Name is what the wire is called by
                the publisher process in its internal operation.
    subscribers: a list of pairs of the form (name, process). Name is what the wire is called by
                the subscriber process in its internal operation.
    shape: a tuple indicating a numpy array shape.
    dtype: a numpy data type for the array elements.
    qtype: a queue type, often RatedSimpleQueue.
    wireList: none or a dictionary with wire names as keys and producer wire objects as values.
Outputs:
    wirelist: an updated wirelist.

IMPORTANT: keep the wire list object for as long as the subprocesses using the wire are running!
The producer object is in the wirelist, and once it is garbage collected, its corresponding shared memory
will be closed and rendered inaccessible to the subprocesses using the wire.

The same process may appear as both publisher and subscriber, however it will never send notifications
to itself (i.e., a subprocess should not use a wire to publish for itself).

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
    producer, consumer = None, None
    if (shape is not None) and (dtype is not None):
        producer, consumer = SHMPort(shape, dtype)
    subscriberQueues = []
    for name, s in subscribers:
        q = qtype()
        subscriberQueues.append((s,q))
        s.addSubscription(name, q, _duplicate(consumer))
    wireList[wireName] = ProducerWire(producer, [x[1] for x in subscriberQueues])
    for name, pub in publishers:
        notifs = [x[1] for x in subscriberQueues if x[0] != pub]
        if 0<len(notifs):
            pub.addPublisher(name, notifs, _duplicate(consumer))
    return wireList

class Peeker:
    def __init__(self):
        self._thread=None
        self._subscriptions={}
        self._data={}
        self._lock=threading.Lock()
        self._work=False
    def addSubscriber(self, name, q, consumer):
        self._subscriptions[name] = (q, consumer)
        self._subscriptions[name] = None
    def _run(self):
        while self._work:
            for name, (q, consumer) in self._subscriptions.items():
                with self._lock:
                    while not q.empty():
                        self._data[name]=q.get()
            time.sleep(0.01)
    def getSubscribedNotification(self, subscriptionName):
        with self._lock:
            return copy.deepcopy(self._data.get(subscriptionName))
    def getSubscribedSHM(self, subscriptionName):
        if (self._subscriptions.get(subscriptionName, [None, None])[1] is not None):
            with self._subscriptions[subscriptionName][1] as consumerSHM:
                a = numpy.copy(consumerSHM)
            return a
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
