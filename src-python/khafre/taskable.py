import ast
from khafre.bricks import RatedSimpleQueue, ReifiedProcess
from multiprocessing import Lock

class TaskableProcess(ReifiedProcess):
    def __init__(self):
        super().__init__()
        self._goalQueue = RatedSimpleQueue()
        self._prefix = None
        self._settings = {}
        self._onSettingsUpdate = {}
        self._queries = set()
        self._queryPredicates = set()
        self._symmetricPredicates = set()
        self._currentGoals = []
    def sendGoal(self, goal):
        self._goalQueue.put(goal)    
    def getGoalQueue(self):
        return self._goalQueue
    def _handleCommand(self, command):
        op, args = command
        if "PUSH_GOALS" == op:
            self.sendGoal(args)
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
            #qobjs = {}
            queries = set()
            queriesToExpand = []
            self._currentGoals = []
            for p,s,o in goals:
                pSplit = p.split("/")[1:]
                if "set" == pSplit[0]:
                    self._update(pSplit[1:], s)
                elif "query" == pSplit[0]:
                    #if p not in qobjs:
                    #    qobjs[p] = set()
                    #qobjs[p].add(s)
                    if o is not None:
                        queries.add(self._orderQuery(p,s,o))
                        #qobjs[p].add(o)
                    else:
                        queries.add((p,s,None))
                        #queriesToExpand.append((p,s))
                else:
                    self._currentGoals.append((p,s,o))
            #for p, s in queriesToExpand:
            #    #_=[queries.add(self._orderQuery(p,s,o)) for o in qobjs[p] if s!=o]
            #    _=[queries.add(self._orderQuery(p,s,o)) for o in self.getQueryUniverseOfDiscourse(p) if s!=o]
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
        self._performStep()
    def _performStep(self):
        pass
