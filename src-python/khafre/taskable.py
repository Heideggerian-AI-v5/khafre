import ast
from khafre.bricks import RatedSimpleQueue, ReifiedProcess
from multiprocessing import Lock


class TriplesFilter:
    def __init__(self, maxDisconfirmations=1, maxNonconfirmations=5):
        self._triples = {}
        self._incompatibles = {}
        self._swapIncompatibles = {}
        self._maxDisconfirmations = maxDisconfirmations
        self._maxNonconfirmations = maxNonconfirmations
    def _negation(self, p):
        if not p.startswith("-"):
            return "-"+p
        return p[1:]
    def _trimTriples(self, k, v):
        toDelete = []
        for e, d in self._triples.items():
            if v <= d[k]:
                toDelete.append(e)
        _ = [self._triples.pop(e) for e in toDelete]
    def setIncompatible(self, p, incompatibles):
        self._incompatibles[p] = self._incompatibles.get(p, set()).union(incompatibles)
        for e in incompatibles:
            self._incompatibles[e] = self._incompatibles.get(e, set()).union([p])
    def setSwapIncompatible(self, p, incompatibles):
        self._swapIncompatibles[p] = self._swapIncompatibles.get(p, set()).union(incompatibles)
        for e in incompatibles:
            self._swapIncompatibles[e] = self._swapIncompatibles.get(e, set()).union([p])
    def setMaxDisconfirmations(self, v):
        self._maxDisconfirmations = v
        self._trimTriples("disconfirmations", v)
    def setMaxNonconfirmations(self, v):
        self._maxNonconfirmations = v
        self._trimTriples("nonconfirmations", v)
    def updateTriples(self):
        toDelete = []
        for t, v in self._triples.items():
            v["nonconfirmations"] += 1
            if self._maxNonconfirmations <= v["nonconfirmations"]:
                toDelete.append(t)
        for e in toDelete:
            self._triples.pop(e)
    def addTriple(self, triple):
        p, s, o = triple
        incompatibles = set(self._incompatibles.get(p, []))
        incompatibles.add(self._negation(p))
        swapIncompatibles = set(self._swapIncompatibles.get(p, []))
        haveIncompatible = False
        for np in incompatibles:
            if (np, s, o) in self._triples:
                self._triples[(np, s, o)]["disconfirmations"] += 1
                if self._maxDisconfirmations <= self._triples[(np, s, o)]["disconfirmations"]:
                    self._triples.pop((np, s, o))
                else:
                    haveIncompatible = True
        for np in swapIncompatibles:
            if (np, o, s) in self._triples:
                self._triples[(np, o, s)]["disconfirmations"] += 1
                if self._maxDisconfirmations <= self._triples[(np, o, s)]["disconfirmations"]:
                    self._triples.pop((np, o, s))
                else:
                    haveIncompatible = True
        if not haveIncompatible:
            self._triples[triple] = {"nonconfirmations": 0, "disconfirmations": 0}
    def getActiveTriples(self):
        return set(self._triples.keys())
    def hasTriple(self, triple):
        return triple in self._triples

class TaskableProcess(ReifiedProcess):
    def __init__(self):
        super().__init__()
        self._goalQueue = RatedSimpleQueue()
        self._prefix = None
        self._settings = {"maxDisconfirmations": 2, "maxNonconfirmations": 30}
        self._onSettingsUpdate = {}
        self._onSettingsUpdate["maxDisconfirmations"] = self._updateMaxDisconfirmations
        self._onSettingsUpdate["maxNonconfirmations"] = self._updateMaxNonconfirmations
        self._queries = set()
        self._queryPredicates = set()
        self._symmetricPredicates = set()
        self._currentGoals = []
        self._triplesFilter = TriplesFilter(maxDisconfirmations=self._settings["maxDisconfirmations"], maxNonconfirmations=self._settings["maxNonconfirmations"])
    def _updateMaxDisconfirmations(self, v):
        self._triplesFilter.setMaxDisconfirmations(v)
    def _updateMaxNonconfirmations(self, v):
        self._triplesFilter.setMaxNonconfirmations(v)
    def sendGoal(self, goal):
        self._goalQueue.put(goal)    
    def getGoalQueue(self):
        return self._goalQueue
    def _customCommand(self, command):
        pass
    def _handleCommand(self, command):
        op, args = command
        if "PUSH_GOALS" == op:
            self.sendGoal(args)
        else:
            self._customCommand(command)
    def _isForMe(self, p):
        return p.startswith(self._prefix+"/")
    def _update(self, address, s):
        cr = self._settings
        for e in address[:-1]:
            if e not in cr:
                cr[e] = {}
            cr = cr[e]
        s = ast.literal_eval(s)
        cr[e] = s
        if address in self._onSettingsUpdate:
            self._onSettingsUpdate[address](s)
    def _orderQuery(self,p,s,o):
        if (p in self._symmetricPredicates) and (s>o):
            return (p, o, s)
        return (p, s, o)
    def _interpretGoal(self):
        if not self._goalQueue.empty():
            goals = [x for x in self._goalQueue.get() if self._isForMe(x[0])]
            queries = set()
            queriesToExpand = []
            self._currentGoals = []
            for p,s,o in goals:
                pSplit = p.split("/")[1:]
                if "set" == pSplit[0]:
                    self._update(pSplit[1:], s)
                elif "query" == pSplit[0]:
                    if o is not None:
                        queries.add(self._orderQuery(p,s,o))
                    else:
                        queries.add((p,s,None))
                else:
                    self._currentGoals.append((p,s,o))
            self._queries = queries
    def _fillInGenericQueries(self, queryUniverseOfDiscourse):
        """
We have two kinds of queries:
    1) specific queries of the form "are X and Y in a particular relationship?" and
    2) generic queries of the form "what is in a particular relation with X?"

The latter are expressed as a triple of form (p, s, None) in the TaskableProcess query list.
This however should be filled in -- converted to a list of specific queries -- at a time
when a "universe of discourse" is known. This universe of discourse is a list of all objects
that could possibly be the answer.
        """
        queries = []
        for q in self._queries:
            if q[2] is not None:
                queries.append(q)
            else:
                _=[queries.append(self._orderQuery(q[0],q[1],o)) for o in queryUniverseOfDiscourse]
        self._queries = queries
    def _doWork(self):
        self._interpretGoal()
        self._triplesFilter.updateTriples()
        self._performStep()
    def _performStep(self):
        pass
