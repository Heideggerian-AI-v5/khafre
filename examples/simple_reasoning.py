import os
import sys
import time

from khafre.bricks import drawWire, setSignalHandlers, startKhafreProcesses, stopKhafreProcesses
from khafre.reasoning import Reasoner

from multiprocessing import Queue
class Peeker:
    def __init__(self):
        self.queue = Queue()
    def sendCommand(self, command):
        self.queue.put(command)

## Reasoning example.

def main():

    triples = [("contact", "cup", "table")]
    percIntTheory = "theories/simple_perceptionInterpretation.dfl"
    updSchTheory = "theories/simple_updateSchemas.dfl"
    connQTheory = "theories/simple_connectivityQueries.dfl"
    schClsTheory = "theories/simple_schemaClosure.dfl"
    schIntTheory = "theories/simple_schemaInterpretation.dfl"
    updQTheory = "theories/simple_updateQueries.dfl"
    backgroundFacts = "theories/simple_backgroundFacts.dfl"
    defaultFacts = "theories/simple_defaultFacts.dfl"
    
    # IMPORTANT: this will be our registry of "wires", connections between khafre subprocesses.
    # Keep this variable alive at least as long as the subprocesses are running.

    wireList = {}
    
    # For ease of start up, cleanup, and setting up termination handlers, process objects should be stored in a dictionary.
    
    procs = {}
    
    procs["reasoner"] = Reasoner()
    outputPeeker = Peeker()
    procs["reasoner"]._workers["outputPeeker"] = outputPeeker
    
    drawWire("TriplesIn", (), [("TriplesIn", procs["reasoner"])], None, None, wireList=wireList)

    sigintHandler, sigtermHandler = setSignalHandlers(procs)
    startKhafreProcesses(procs)

    procs["reasoner"].sendCommand(("LOAD_THEORY", ("perception interpretation", percIntTheory)))
    procs["reasoner"].sendCommand(("LOAD_THEORY", ("update schemas", updSchTheory)))
    procs["reasoner"].sendCommand(("LOAD_THEORY", ("connectivity queries", connQTheory)))
    procs["reasoner"].sendCommand(("LOAD_THEORY", ("schema closure", schClsTheory)))
    procs["reasoner"].sendCommand(("LOAD_THEORY", ("schema interpretation", schIntTheory)))
    procs["reasoner"].sendCommand(("LOAD_THEORY", ("update questions", updQTheory)))
    procs["reasoner"].sendCommand(("LOAD_FACTS", (backgroundFacts,)))
    #procs["reasoner"].sendCommand(("REGISTER_WORKER", (outputPeeker,)))
    procs["reasoner"].sendCommand(("TRIGGER", (defaultFacts,)))

    while outputPeeker.queue.empty():
        time.sleep(0.1)

    print("Initial results", outputPeeker.queue.get())
    
    if wireList["TriplesIn"].isReadyForPublishing():
        print("Push triples", triples)
        wireList["TriplesIn"].publish({"triples": triples}, None)

    while outputPeeker.queue.empty():
        time.sleep(0.1)

    print("Second results", outputPeeker.queue.get())

    stopKhafreProcesses(procs)

if "__main__" == __name__:
    main()
