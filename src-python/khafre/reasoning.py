'''
"reasoning":
    - gets triples and other outputs from taskables
    - sends commands to taskables
    - outline of process:
        - perception interpretation: observations (tuples) + previous persistent schemas (dfl facts) + theory -> reifiable relations, stet relations (triples)
        - update schemas: reifiable/stet relations (triples) + theory -> new persistent schemas (dfl facts)
        - schema interpretation p1: new persistent schemas (dfl facts) + theory -> connquery graph edges (triples)
        - connection query: connquery -> connquery results (triples)
        - add connquery results to persistent schemas
        - schema interpretation p2: new persistent schemas (dfl facts) + theory -> reifiable questions, stet relations (triples)
        - question update: reifiable questions/stet relations + theory -> new questions (tuples)
annotated image saver
'''

import cv2 as cv
import itertools
import json
import networkx
import numpy
import os
import time

from PIL import Image

from khafre.bricks import ReifiedProcess
from khafre.polygons import findTopPolygons
import khafre.silkie as silkie

# Graph connectivity queries

def closenessQuery(graph, source, target):
    """
    If source and target are connected: returns a list of nodes that are closer to the target than source.
    If no path, returns [].
    """
    try:
        lengths = networkx.shortest_path_length(graph, target=target)
    except networkx.NodeNotFound:
        return []
    if source not in lengths:
        return []
    return [k for k,v in lengths.items() if v < lengths[source]] + [source]

def necessaryVertexQuery(graph, source, target):
    """
    If source and target are connected: returns a list of nodes that all paths from source to target must pass through.
    If source and target are not connected, returns [].
    """
    def _weight(bigWeight, forbiddenV, startV, targetV, attributes):
        if forbiddenV == startV:
            return bigWeight
        return 1
    try:
        networkx.shortest_path_length(graph, source=source, target=target)
    except networkx.exception.NetworkXNoPath:
        return []
    N = graph.number_of_nodes()
    retq = []
    for e in graph.nodes:
        if (e != source) and (e != target):
            if N <= networkx.shortest_path_length(graph, source=source, target=target, weight=(lambda s,t,a: _weight(N,e,s,t,a))):
                retq.append(e)
    return retq

# Silkie interfaces
def copyFacts(a):
    '''
    Return a dictionary (predicate name: associated PFact) that is a copy of another dictionary of silkie PFacts. 
    '''
    retq = {k: silkie.PFact(k) for k in a}
    for k in retq:
        _ = [retq[k].addFact(t[1], t[2], silkie.DEFEASIBLE) for t in a[k].getTriples()]
    return retq

def mergeFacts(a,b):
    '''
    Given two dictionaries of silkie PFacts, merges the second into the first.
    '''
    for k in b:
        if k not in a:
            a[k] = silkie.PFact(k)
        _ = [a[k].addFact(t[1], t[2], silkie.DEFEASIBLE) for t in b[k].getTriples()]
    return a

def triples2Facts_internal(ts, prefix=None):
    '''
    Converts a list of triples into a dictionary of silkie PFacts
    '''
    if prefix is None:
        prefix = ""
    retq = {}
    for t in ts:
        p,s,o = t
        if p.startswith("-"):
            p = "-" + prefix + p[1:]
        else:
            p = prefix + p
        if p not in retq:
            retq[p] = silkie.PFact(p)
        retq[p].addFact(s, o, silkie.DEFEASIBLE)
    return retq

def triples2Facts(ts, prefix=None):
    '''
    Converts a list or dictionary of triples into a dictionary of silkie PFacts
    '''
    if isinstance(ts,dict):
        retq = {}
        for k,v in ts.items():
            nretq = triples2Facts_internal(v,prefix)
            for nk, nv in nretq.items():
                if nk not in retq:
                    retq[nk] = silkie.PFact(nk)
                _ = [retq[nk].addFact(t[1], t[2], silkie.DEFEASIBLE) for t in nv.getTriples()]
        return retq
    else:
        return triples2Facts_internal(ts,prefix)

def conclusions2Facts(conclusions):
    '''
    Converts the defeasibly provable conclusions of a silkie result into a dictionary of silkie PFacts.
    '''
    return triples2Facts(conclusions.defeasiblyProvable)

def conclusions2Graph(conclusions, edgePredicate=None):
    '''
    Converts conclusions into a networkx.DiGraph where edges appear between entities participating in a given predicate
    '''
    if edgePredicate is None:
        edgePredicate = "connQueryEdge"
    retq = networkx.DiGraph()
    _ = [retq.add_edge(t[1],t[2]) for t in conclusions.defeasiblyProvable if edgePredicate == t[0]]
    return retq

def reifyConclusions(conclusions):
    '''
    Prepare a list of statements from stets and reifiables.
    '''
    def _dropStet(x):
        if '-' == x[0]:
            return '-' + x[len('-stet_'):]
        return x[len("stet_"):]
    stets = [(_dropStet(t[0]), t[1], t[2]) for t in conclusions.defeasiblyProvable if (t[0].startswith("stet_")) or (t[0].startswith("-stet_"))]
    reifiables = sorted([t for t in conclusions.defeasiblyProvable if t[0].startswith("reifiable_")])
    reifieds = []
    for k,r in enumerate(reifiables):
        p, s, o = r
        p = p[len("reifiable_"):]
        reification = "reification%d" % k
        reifieds += [("isA", reification, p), ("hasS", reification, s), ("hasO", reification, o)]
    return triples2Facts(stets + reifieds)


#def queryDesc2Facts(qd):
#    source, target, _, _, qtype = qd
#    triples = [("isA", "runningQuery", qtype), ("hasSource", "runningQuery", source), ("hasTarget", "runningQuery", target)]
#    return triples2Facts(triples)

class Reasoner(ReifiedProcess):
    def __init__(self):
        super().__init__()
        self.perceptionInterpretationTheory = silkie.TheoryTemplate()
        self.updateSchemasTheory = silkie.TheoryTemplate()
        self.connQueryTheory = silkie.TheoryTemplate()
        self.closureTheory = silkie.TheoryTemplate()
        self.schemaInterpretationTheory = silkie.TheoryTemplate()
        self.updateQuestionsTheory = silkie.TheoryTemplate()
        self.backgroundFacts = {}
        self.persistentSchemas = {}
        self.perceptionQueries = []
        self._workers = {}
        self._armed = False
        self._eventPath = None
        self._previousSchemaTriples = []
        self._defaultFacts = {}
        self._storageMap = {}
    def _internalEvent(self):
        return self._armed
    def _checkPublisherRequest(self, name, wire):
        return True#name in self._outputNames
    def _checkSubscriptionRequest(self, name, wire):
        return True
    def _handleCommand(self, command):
        op, args = command
        # TODO: consider splitting reasoning pipeline across several bricks, to allow for variability in its contents.
        # Would need: 
        #    rest-of-khafre to Silkie, Silkie to Silkie, Silkie to rest-of-khafre connections
        #    reification/stets brick
        #    connectivity query brick
        #    memory block for persistent schemas? embed state memory in reasoner block?
        if "SET_PATH" == op:
            self._eventPath = args[0]
        elif "TRIGGER" == op:
            self._armed = True
            self._defaultFacts = silkie.loadDFLFacts(args[0])
        elif "RESET_SCHEMAS" == op:
            self.persistentSchemas = {}
        elif "REGISTER_STORAGE_DESTINATION" == op:
            inputName, outputName = args
            self._storageMap[inputName] = outputName
        elif "LOAD_THEORY" == op:
            fn, path = args
            if "perception interpretation" == fn:
                self.perceptionInterpretationTheory = silkie.loadDFLRules(path)
            elif "update schemas" == fn:
                self.updateSchemasTheory = silkie.loadDFLRules(path)
            elif "connectivity queries" == fn:
                self.connQueryTheory = silkie.loadDFLRules(path)
            elif "schema closure" == fn:
                self.closureTheory = silkie.loadDFLRules(path)
            elif "schema interpretation" == fn:
                self.schemaInterpretationTheory = silkie.loadDFLRules(path)
            elif "update questions" == fn:
                self.updateQuestionsTheory = silkie.loadDFLRules(path)
            elif "interpret masks" == fn:
                self.interpretMasksTheory = silkie.loadDFLRules(path)
            elif "store masks" == fn:
                self.storeMasksTheory = silkie.loadDFLRules(path)
        elif "LOAD_FACTS" == op:
            path = args[0]
            self.backgroundFacts = silkie.loadDFLFacts(path)
    def _doWork(self):
        def _getSchemas(triples):
            def _summary(statement):
                retq = []
                if "isA" in statement:
                    retq.append(("type", tuple(sorted(list(statement["isA"])))))
                for k, v in statement.items():
                    if "isA" != k:
                        retq.append((k, tuple(sorted(list(v)))))
                return tuple(sorted(retq))
            loggables = [t[1] for t in triples if ("isA" == t[0]) and ("Loggable" == t[2])]
            statements = {k: {} for k in loggables}
            participants = set()
            for t in triples:
                if t[1] in statements:
                    if ("isA" == t[0]) and ("Loggable" != t[2]):
                        statements[t[1]]["type"] = statements[t[1]].get("type", set()).union([t[2]])
                    elif ("isA" != t[0]) and (t[0] not in ["hasO", "hasS"]):
                        statements[t[1]][t[0]] = statements[t[1]].get(t[0], set()).union([t[2]])
                        participants.add(t[2])
            participantTypes = {}
            for t in triples:
                if ("isA" == t[0]) and (t[1] in participants):
                    participantTypes[t[1]] = participantTypes.get(t[1], set()).union([t[2]])
            participants = sorted(list(participants))
            retq = ([(("name", p), ("type", tuple(sorted(list(participantTypes.get(p, [])))))) for p in participants])
            retq += sorted([_summary(x) for x in statements.values()])
            retq = [x for x in retq if () != x]
            return set(retq)
        def _ensureTriple(t):
            if (3 == len(t)) and ("" != t[2]):
                return tuple(t)
            return (t[0], t[1], None)
        def _silkie(theoryTemplate, facts, backgroundFacts):
            theory, _, i2s, _ = silkie.buildTheory(theoryTemplate, facts, backgroundFacts)
            return silkie.idx2strConclusions(silkie.dflInference(theory), i2s)
        def _prepareMaskResults(masksToStore, triples, images, maskPolygons, previous):
            retq = previous
            if retq is None:
                retq = {}
            idxMap = {}
            aboutMap = {}
            fromMap = {}
            ignoreMasks = set()
            maskNames = set()
            for t in triples:
                for p, d in [("about", aboutMap), ("hasId", idxMap), ("from", fromMap)]:
                    if p == t[0]:
                        if t[1] in d:
                            ignoreMasks.add(t[1])
                        d[t[1]] = t[2]
                if "hasConjunct" == t[0]:
                    maskNames.add(t[1]), maskNames.add(t[2])
            maskNames = maskNames.union(masksToStore)
            for m in maskNames:
                if (m not in fromMap) or (m not in idxMap) or (m not in aboutMap) or (aboutMap[m] not in images) or (fromMap[m] not in images):
                    ignoreMasks.add(m)
            for m in masksToStore:
                if m in ignoreMasks:
                    continue
                maskTriples = [t for t in triples if m == t[1]]
                maskKnowledge = {}
                for t in maskTriples:
                    if t[0] not in maskKnowledge:
                        maskKnowledge[t[0]] = set()
                    maskKnowledge[t[0]].add(t[2])
                sourceImg = list(maskKnowledge["from"])[0]
                aboutImg = list(maskKnowledge["about"])[0]
                aux = {"semantics": {k: maskKnowledge.get(k, set()) for k in ["masksPartOfObjectType", "usedForTaskType", "playsRoleType"]}}
                if "hasConjunct" in maskKnowledge:
                    maskImg = numpy.ones(images[sourceImg].shape, dtype=numpy.uint8)*255
                    for conjunct in maskKnowledge["hasConjunct"]:
                        if (conjunct not in ignoreMasks) and (aboutMap[conjunct] == aboutMap[m]):
                            maskImg[images[fromMap[conjunct]] != idxMap[conjunct]] = 0
                    if 2 != len(numpy.unique(maskImg)):
                        continue
                    aux["polygons"] = findTopPolygons(maskImg)
                elif ("hasId" in maskKnowledge) and (1 == len(maskKnowledge["hasId"])):
                    idx = int(list(maskKnowledge["hasId"])[0])
                    aux["polygons"] = maskPolygons[idx]
                else:
                    continue
                if aboutImg not in retq:
                    retq[aboutImg] = []
                retq[aboutImg].append(aux)
            return retq
        # Because of bypass event, the reasoner will reach here without waiting in the reified process main loop.
        # By this point, any received commands have been handled and any ready to send outputs were sent.
        # A full set of inputs might not be available however.
        fullInput = all([v.get("notification") is not None for v in self._dataFromSubscriptions.values()])
        maskPolygons = {}
        inpImgId = None
        if self._armed:
            facts = mergeFacts(self.persistentSchemas, self._defaultFacts)
            maskFacts = []
            imageResources = {}
            isAs = []
        elif fullInput:
            # TODOs for TASKABLES:
            #    - let optical flow report movement masks
            #    - movement masks associated to triples
            # Collect all inputs into an initial theory
            triples = set.union(*[set(v["notification"].get("triples", [])) for v in self._dataFromSubscriptions.values()])
            isAs = [t for t in triples if "isA" == t[0]]
            facts = mergeFacts(self.persistentSchemas, triples2Facts(triples))
            imageResources = {k: v.get("image") for k, v in self._dataFromSubscriptions.items()}
            inpImgId = self._dataFromSubscriptions.get("InpImg",{}).get("notification")
            if inpImgId is None:
                inpImgId = time.perf_counter()
            else:
                if isinstance(inpImgId, str):
                    inpImgId = json.loads(inpImgId)
                inpImgId = inpImgId.get("imgId", time.perf_counter())
            maskFacts = []
            cr = 0
            for k in self._dataFromSubscriptions.keys():
                for m in self._dataFromSubscriptions[k]["notification"].get("masks", []):
                    name = "mask_%d" % cr
                    cr += 1
                    maskFacts.append(("isA", name, "ImageMask"))
                    maskFacts.append(("from", name, k))
                    maskFacts.append(("hasId", name, m["hasId"]))
                    maskFacts.append(("hasP", name, m["hasP"]))
                    maskFacts.append(("hasS", name, m["hasS"]))
                    maskFacts.append(("hasO", name, m["hasO"]))
                    maskPolygons[int(m["hasId"])] = m["polygons"]
        if fullInput or self._armed:
            self._armed = False
            for k in self._dataFromSubscriptions.keys():
                self._dataFromSubscriptions[k] = {"notification": None}
            ## observations (tuples) + prev persistent schemas (dfl facts) + theory -> reifiable/stet relations (triples)
            conclusions = _silkie(self.perceptionInterpretationTheory, facts, self.backgroundFacts)
            ## reifiable/stet relations (triples) + theory -> new persistent schemas (dfl facts)
            facts = reifyConclusions(conclusions)
            conclusions = _silkie(self.updateSchemasTheory, facts, self.backgroundFacts)
            self.persistentSchemas = conclusions2Facts(conclusions)
            ## add connquery results to persistent schemas
            connectivityResults = {}
            connectivityQueries = {}
            for t in self.perceptionQueries:
                p, s, o = t
                if ("isA" == p) and (o.startswith("connectivity/")):
                    connectivityQueries[s] = {}
            for t in self.perceptionQueries:
                p, s, o = t
                if s in connectivityQueries:
                    if p not in connectivityQueries[s]:
                        connectivityQueries[s][p] = set()
                    connectivityQueries[s][p].add(o)
            for name, query in connectivityQueries.items():
                for qtype, fn in [("connectivity/closeness", closenessQuery), ("connectivity/dependency", necessaryVertexQuery)]:
                    if qtype not in query["isA"]:
                        continue
                    srcTrgPairs = itertools.product(query["hasSource"], query["hasTarget"])
                    psPairs = itertools.product(query["hasP"], query["hasQS"])
                    for source, target in srcTrgPairs:
                        qTriples = [("hasSource", name, source), ("hasTarget", name, target)]
                        _ = [qTriples.append(("isA", name, x) for x in query["isA"])]
                        facts = mergeFacts(copyFacts(persistentSchemas), triples2Facts(qTriples))
                        conclusions = _silkie(self.connQueryTheory, facts, self.backgroundFacts)
                        graph = conclusions2Graph(conclusions)
                        fnResults = fn(graph, source, target)
                        for p, s in psPairs:
                            triples = [(p, s, e) for e in fnResults]
                            connectivityResults = mergeFacts(connectivityResults, triples2Facts(triples))
            self.persistentSchemas = mergeFacts(self.persistentSchemas, connectivityResults)
            conclusions = _silkie(self.closureTheory, self.persistentSchemas, self.backgroundFacts)
            if fullInput and (self._eventPath is not None) and (inpImgId is not None):
                schemasNow = _getSchemas(conclusions.defeasiblyProvable)
                schemasPrev= _getSchemas(self._previousSchemaTriples)
                added = schemasNow.difference(schemasPrev)
                lost = schemasPrev.difference(schemasNow)
                toLog = {}
                if 0 != len(added):
                    toLog["added"] = sorted(list(added))
                if 0 != len(lost):
                    toLog["lost"] = sorted(list(lost))
                if 0 != len(toLog):
                    toLog["image_id"] = inpImgId
                    fnamePrefix = os.path.join(self._eventPath, "evt_%s" % str(time.perf_counter()))
                    if "InpImg" in imageResources:
                        imageBGR = cv.cvtColor(imageResources["InpImg"], cv.COLOR_BGR2RGB)
                        Image.fromarray(imageBGR).save(fnamePrefix + ".jpg")
                    with open(fnamePrefix + ".json", "w") as outfile:
                        _ = outfile.write("%s\n" % json.dumps(toLog))
            self._previousSchemaTriples = set(conclusions.defeasiblyProvable)
            self.persistentSchemas = conclusions2Facts(conclusions)            
            ## new persistent schemas (dfl facts) + theory -> reifiable questions, stet relations (triples)
            conclusions = _silkie(self.schemaInterpretationTheory, self.persistentSchemas, self.backgroundFacts)
            ## reifiable questions/stet relations + theory -> new questions (tuples)
            facts = reifyConclusions(conclusions)
            conclusions = _silkie(self.updateQuestionsTheory, facts, self.backgroundFacts)
            self.perceptionQueries = [_ensureTriple(t) for t in conclusions.defeasiblyProvable]
            ## masks to store / reified conjunctions of masks
            facts = mergeFacts(self.persistentSchemas, triples2Facts(maskFacts))
            conclusions = _silkie(self.interpretMasksTheory, facts, self.backgroundFacts)
            masksToStore = [t[1] for t in conclusions.defeasiblyProvable if "storeMask" == t[0]]
            maskResults = _prepareMaskResults(masksToStore, conclusions.defeasiblyProvable, imageResources, maskPolygons, None)
            ## constructed masks to store
            facts = reifyConclusions(conclusions)
            conclusions = _silkie(self.storeMasksTheory, facts, self.backgroundFacts)
            masksToStore = [t[1] for t in conclusions.defeasiblyProvable if "storeMask" == t[0]]
            maskResults = _prepareMaskResults(masksToStore, conclusions.defeasiblyProvable, imageResources, maskPolygons, maskResults)
            ## send masks to store
            for inputName, results in maskResults.items():
                if (inputName in self._storageMap) and (inputName in self._dataFromSubscriptions):
                    self._requestToPublish(self._storageMap[inputName], results, imageResources[inputName])
            ## send perception queries
            _ = [self._callWorker(x, ("PUSH_GOALS", self.perceptionQueries)) for x in self._workers.values()]
    def registerWorker(self, name, proc):
        self._workers[name] = (proc._command, proc._event)
    def _callWorker(self, worker, command, block=False, timeout=None):
        q, e = worker
        q.put(command)#, block=block, timeout=timeout)
        with e:
            e.notify_all()

