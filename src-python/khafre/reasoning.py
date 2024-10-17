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

import itertools
import networkx
import os
import time

from khafre.bricks import ReifiedProcess
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
        #print("NF")
        return []
    if source not in lengths:
        #print("NL")
        return []
    #print(lengths)
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
    #_ = [print((t[1],t[2])) for t in conclusions.defeasiblyProvable if edgePredicate == t[0]]
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
        self._dbgImg = None
        self._bypassEvent = True
        #self._outputNames = {"OutImg", "DbgImg"}
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
    def _checkPublisherRequest(self, name, wire):
        return False#name in self._outputNames
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
        if "REGISTER_WORKER" == op:
            name, proc = args
            self._workers[name] = proc
        elif "UNREGISTER_WORKER" == op:
            name = args[0]
            if name in self._workers:
                self._workers.pop(name)
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
            elif "schema interpretation":
                self.schemaInterpretationTheory = silkie.loadDFLRules(path)
            elif "update questions" == fn:
                self.updateQuestionsTheory = silkie.loadDFLRules(path)
        elif "LOAD_FACTS" == op:
            path = args[0]
            self.backgroundFacts = silkie.loadDFLFacts(path)
    def _doWork(self):
        def _ensureTriple(t):
            if 3 == len(t):
                return tuple(t)
            return (t[0], t[1], None)
        def _silkie(theoryTemplate, facts, backgroundFacts):
            theory, _i2s, _ = silkie.buildTheory(theoryTemplate, facts, backgroundFacts)
            return silkie.idx2strConclusions(silkie.dflInference(theory), i2s)
        # Because of bypass event, the reasoner will reach here without waiting in the reified process main loop.
        # By this point, any received commands have been handled and any ready to send outputs were sent.
        # A full set of inputs might not be available however.
        fullInput = all([v is not None for v in self._dataFromSubscriptions.values()])
        if fullInput:
            # TODOs for TASKABLES:
            #    - let optical flow report movement masks
            #    - contact/movement masks associated to triples
            # Collect all inputs into an initial theory
            # TODO: use images too for grounding mask entities
            triples = set.union([v["notification"].get("triples", []) for v in self._dataFromSubscriptions.values()])
            ## observations (tuples) + prev persistent schemas (dfl facts) + theory -> reifiable/stet relations (triples)
            facts = mergeFacts(self.persistentSchemas, triples2Facts(triples))
            conclusions = _silkie(self.perceptionInterpretationTheory, facts, self.backgroundFacts)
            ## reifiable/stet relations (triples) + theory -> new persistent schemas (dfl facts)
            facts = reifyConclusions(conclusions)
            conclusions = _silkie(self.updateSchemasTheory, facts, self.backgroundFacts)
            self.persistentSchemas = conclusions2Facts(conclusions)
            for k in self._dataFromSubscriptions.keys():
                self._dataFromSubscriptions[k] = None
        #else:
        #    # Keep persistent schemas, send commands/outputs based on those.
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
        self.persistentSchemas = conclusions2Facts(conclusions)
        ## new persistent schemas (dfl facts) + theory -> reifiable questions, stet relations (triples)
        conclusions = _silkie(self.schemaInterpretationTheory, self.persistentSchemas, self.backgroundFacts)
        ## reifiable questions/stet relations + theory -> new questions (tuples)
        facts = reifyConclusions(conclusions)
        conclusions = _silkie(self.updateQuestionsTheory, self.facts, self.backgroundFacts)
        self.perceptionQueries = [_ensureTriple(t) for t in conclusions.defeasiblyProvable]
        ## send perception queries
        _ = [x.sendCommand("PUSH_GOALS", self.perceptionQueries) for x in self._workers.values()]
