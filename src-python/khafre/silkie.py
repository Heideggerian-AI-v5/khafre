# DFL Operator
# The core:
#     <+Fr> <- [+Dq] # r is factual if all of its antecedents are +D
#     [+Dq] <- <+Fr> # q is strictly provable if some factual rule r has it as consequent
#     [-Fr] <- <-Dq> # r is demoted if some of its antecedents are -D # NOTE: will not need this explicitly stored
#     <-Dq> <- [-Fr] # q is strictly unprovable if all rules with it as consequent are demoted
#     <+Ar> <- [+dq] # a rule is applicable if all its antecedents are +d
#     [-Ar] <- <-dq> # a rule is inapplicable if some of its antecedents are -d
#     [+dq] <- [+Dq] # q is defeasibly provable if it is strictly provable
#     [+dq] <- <+Tr> # q is defeasibly provable if some triggered rule r has it as consequent
#     <-dq> <- <-Dq> # q is defeasibly unprovable only if it is strictly unprovable
#     <-dq> <- [-Tr] # q is defeasibly unprovable only if all rules with it as consequent are cancelled
#     <+Tr> <- <-Dq> # r is triggered only if its opposing consequent is not strictly provable
#     <+Tr> <- [+dq] # r is triggered only if all its antecedents are defeasibly provable
#     [-Tr] <- [+Dq] # a rule is cancelled if it opposes a strictly proven literal
#     [-Tr] <- <-dq> # a rule is cancelled if one of its antecedents is -d
# The defeasible extension: no team defeat
#     <+Tr> <- [-Ws] # a rule is triggered only if all rules opposing it that it does not dominate are inactive
#     [-Tr] <- <+Ws> # a rule is cancelled if some opposing rule it does not dominate is active
# The defeasible extension: team defeat
#     <+Tr> <- [-Bs] # r is triggered only if all rules s opposing it are blocked
#     [-Tr] <- <+Bs> # a rule is cancelled if a rule s opposing it is blocking
#     [-Br] <- <+Ws> # a rule is blocked if it is dominated by an active rule s
#     <+Br> <- [-Ws] # a rule is blocking only if all rules dominating it are inactive
# The defeasible extension: ambiguity blocking
#     <+Ws> <- <+Ar> # a rule is active if it is applicable (there are no other edges towards <+Ws> anyway)
#     [-Ws] <- [-Ar] # a rule is inactive if it is inapplicable
#     [-Br] <- <-dq> # a rule is blocked if some of its antecedents are -d
#     <+Br> <- [+dq] # a rule is blocking only if all its antecedents are +d
# The defeasible extension: ambiguity propagation
#     <+Ur> <- [+Sq] # a rule is supported only if all its antecedents are +S
#     [-Ur] <- <-Sq> # a rule is unsupported if some of its antecedents are -S
#     <+Wr> <- <+Ur> # a rule is active if it is supported (there are no other edges towards <+Wr> anyway)
#     [-Wr] <- [-Ur] # a rule is inactive if it is unsupported
#     [-Br] <- <-Sq> # a rule is blocked if some of its antecedents are -S
#     <+Br> <- [+Sq] # a rule is blocking only if all its antecedents are +S
#     [+Sq] <- [+Dq] # q is supported if it is strictly provable
#     <-Sq> <- <-Dq> # q is unsupported only if it is not strictly provable
#     [+Sq] <- <+Pr> # q is supported if some rule with it as consequent is plausible
#     <+Pr> <- <-D-q> # a rule is plausible only if its consequent is not strictly opposed
#     <+Pr> <- [+Sq] # a rule is plausible only if all its antecedents are +S
#     <+Pr> <- [-As] # a rule is plausible only if all rules s dominating it are inapplicable # NOTE: there seems to be a typo in the DFL Families paper here
#     <-Sq> <- [-Pr] # q is unsupported if all rules with it as consequent are implausible
#     [-Pr] <- [+D-q] # a rule is implausible if its consequent is strictly opposed
#     [-Pr] <- <-Sq> # a rule is implausible if some of its antecedents are -S
#     [-Pr] <- <+As> # a rule is implausible if some rule s dominating it is applicable
#
# The well-foundedness operator
# The strict core
#     [+Dq] <- <+Fr> # a literal is +D when a strict rule with it as consequent is factual
#     <+Fr> <- [+Dq] # a rule is factual only if all its antecedents are +D
#     anything left out of the lfp of this op is -D or -F
# Note: for strict inference we don't need anything special, as the strict conclusions can be obtained in one call of lfp for the inference op.
# The defeasible core
#     <fq> <- [rq] # a literal is founded only when when it is foundation-reachable
#     <fq> <- [nq] # a literal is founded only when it is NOT proven -dq
#     [rq] <- <er> # a literal is foundation-reachable when it has an established rule
#                  # a literal is also foundation-reachable when it is proven +dq
#     <er> <- <fq> # a rule is established when all its antecedents are founded
#     [nq] <- empty # set inside the attractor if and only if -dq not proven

# Operator flags
STRICT = 1
DEFEASIBLE = 2
DEFEATER = 3
class Rule:
    def __init__(self):
        self.id = 0
        self.operator = 0
        self.antecedent = set()
        self.consequent = 0
        self.dominates = set()
class Conclusions:
    def __init__(self):
        self.strictlyProvable = set()
        self.strictlyUnprovable = set()
        self.defeasiblyProvable = set()
        self.defeasiblyUnprovable = set()
        self.supportable = set()
        self.unsupportable = set()
        self.contradictions = set()
class Vertex:
    def __init__(self,dbg=None):
        self.dbg=dbg
        self.enabled = True
        self.inverts = []
        self.inAttractor = False
    def _bracket(self):
        return "(%s)"
    def __repr__(self):
        return str(self)
    def __str__(self):
        return self._bracket() % str(self.dbg)
    def _onAddedAsInVert(self):
        return
    def addInVert(self, vertex):
        self.inverts.append(vertex)
        vertex._onAddedAsInVert()
    def fallIn(self, events):
        self.inAttractor = True
        for iv in self.inverts:
            events.append(iv)
        return events
class EvenVertex(Vertex):
    def __init__(self,dbg=None):
        super().__init__(dbg)
    def _bracket(self):
        return "[%s]"
class OddVertex(Vertex):
    def __init__(self,dbg=None):
        super().__init__(dbg)
        self.safeOutverts = 0
        self.inAttractor = True
    def _bracket(self):
        return "<%s>"
    def _onAddedAsInVert(self):
        self.inAttractor = False
        self.safeOutverts += 1

def attractor(verts):
    events = []
    retq = False
    #print("Attractor fallIn")
    for v in verts:
        if v.inAttractor:
            #print(str(v))
            v.fallIn(events)
    #print("Attractor events")
    while events:
        v = events.pop()
        #print(str(v))
        if not v.inAttractor:
            if isinstance(v, EvenVertex):
                retq = True
                v.fallIn(events)
                #print("\tcap")
            elif isinstance(v, OddVertex):
                v.safeOutverts -= 1
                if 0 == v.safeOutverts:
                    retq = True
                    v.fallIn(events)
                    #print("\tcap")
    return retq

def opposesAssertion(q, r):
    return (((q == -r.consequent) and (not (DEFEATER == r.operator))) or ((q == r.consequent) and (DEFEATER == rj.operator)))

def dflInference(theory, teamDefeat=True, ambiguityPropagation=True, wellFoundedness=True):
    retq = Conclusions()
    # Stage 0: construct maps from theory entities to vertices in the inference graphs
    verts = 0
    literals = set()
    for x in theory:
        literals.add(x.consequent)
        literals.add(-x.consequent)
        for a in x.antecedent:
            literals.add(a)
            literals.add(-a)
    rules = set([x.id for x in theory])
    strictRules = set([x.id for x in theory if STRICT == x.operator])
    opponents = {l: [] for l in literals}
    supporters = {l: []  for l in literals}
    for r in theory:
        if DEFEATER == r.operator:
            opponents[r.consequent].append(r)
        else:
            opponents[-r.consequent].append(r)
            supporters[r.consequent].append(r)
    pD = {l: EvenVertex("+D%d"%l) for l in literals} # [+D]
    mD = {l: OddVertex("-D%d"%l) for l in literals} # <-D>
    pd = {l: EvenVertex("+d%d"%l) for l in literals} # [+d]
    md = {l: OddVertex("-d%d"%l) for l in literals} # <-d>
    pF = {r: OddVertex("+F%d"%r) for r in strictRules} # <+F>
    pA = {r: OddVertex("+A%d"%r) for r in rules} # <+A>
    mA = {r: EvenVertex("-A%d"%r) for r in rules} # [-A]
    pW = {r: OddVertex("+W%d"%r) for r in rules} # <+W>
    mW = {r: EvenVertex("-W%d"%r) for r in rules} # [-W]
    pT = {r: OddVertex("+T%d"%r) for r in rules} # <+T>
    mT = {r: EvenVertex("-T%d"%r) for r in rules} # [-T]
    pB = {r: OddVertex("+B%d"%r) for r in rules} # <+B>
    mB = {r: EvenVertex("-B%d"%r) for r in rules} # [-B]
    if ambiguityPropagation:
        pS = {l: EvenVertex("+S%d"%l) for l in literals} # [+S]
        mS = {l: OddVertex("-S%d"%l) for l in literals} # <-S>
        pP = {r: OddVertex("+P%d"%r) for r in rules} # <+P>
        mP = {r: EvenVertex("-P%d"%r) for r in rules} # [-P]
        pU = {r: OddVertex("+U%d"%r) for r in rules} # <+U>
        mU = {r: EvenVertex("-U%d"%r) for r in rules} # [-U]
    if wellFoundedness:
        x = {}
    # Stage 1: construct the inference graph for well-founded strict inference and run lfp of operator
    strictInference = [pD[l] for l in pD.keys()] + [pF[r] for r in pF.keys()]
    for r in theory:
        if STRICT == r.operator:
            pF[r.id].addInVert(pD[r.consequent])
            for a in r.antecedent:
                pD[a].addInVert(pF[r.id])
    attractor(strictInference)
    for l in literals:
        if pD[l].inAttractor:
            retq.strictlyProvable.add(l)
            if pD[-l].inAttractor:
                retq.contradictions.add(l)
            mD[l].inAttractor = False
            pD[l].inverts = [] # Will reuse this vertex in the defeasible inference graph, but no need to waste time propagating the attractor
            # to <+F> vertices anymore, because they won't be used again; we'll add new inverts
        else:
            retq.strictlyUnprovable.add(l)
            mD[l].inAttractor = True
    # Stage 2: construct inference graphs for defeasible inference
    for r in theory:
        for a in r.antecedent:
    #     <+Ar> <- [+dq] # a rule is applicable if all its antecedents are +d
            pd[a].addInVert(pA[r.id])
    #     [-Ar] <- <-dq> # a rule is inapplicable if some of its antecedents are -d
            md[a].addInVert(mA[r.id])
            if not (DEFEATER == r.operator):
    #     <+Tr> <- [+dq] # r is triggered only if all its antecedents are defeasibly provable
                pd[a].addInVert(pT[r.id])
    #     [-Tr] <- <-dq> # a rule is cancelled if one of its antecedents is -d
                md[a].addInVert(mT[r.id])
        if not (DEFEATER == r.operator):
    #     [+dq] <- <+Tr> # q is defeasibly provable if some triggered rule r has it as consequent
            pT[r.id].addInVert(pd[r.consequent])
    #     <-dq> <- [-Tr] # q is defeasibly unprovable only if all rules with it as consequent are cancelled
            mT[r.id].addInVert(md[r.consequent])
    #     <+Tr> <- <-Dq> # r is triggered only if its opposing consequent is not strictly provable
            mD[-r.consequent].addInVert(pT[r.id])
    #     [-Tr] <- [+Dq] # a rule is cancelled if it opposes a strictly proven literal
            pD[-r.consequent].addInVert(mT[r.id])
    for l in pd.keys():
    #     [+dq] <- [+Dq] # q is defeasibly provable if it is strictly provable
        pD[l].addInVert(pd[l])
    #     <-dq> <- <-Dq> # q is defeasibly unprovable only if it is strictly unprovable
        mD[l].addInVert(md[l])
    if not teamDefeat:
        for rk in theory:
            if not (DEFEATER == rk.operator):
                for rj in opponents[rk.consequent]:
                    if (rj.id not in rk.dominates):
    #     <+Tr> <- [-Ws] # a rule is triggered only if all rules opposing it that it does not dominate are inactive
                        mW[rj.id].addInVert(pT[rk.id])
    #     [-Tr] <- <+Ws> # a rule is cancelled if some opposing rule it does not dominate is active
                        pW[rj.id].addInVert(mT[rk.id])
    else:
        for rk in theory:
            if not (DEFEATER == rk.operator):
                for rj in opponents[rk.consequent]:
    #     <+Tr> <- [-Bs] # r is triggered only if all rules s opposing it are blocked
                    mB[rj.id].addInVert(pT[rk.id])
    #     [-Tr] <- <+Bs> # a rule is cancelled if a rule s opposing it is blocking
                    pB[rj.id].addInVert(mT[rk.id])
                    if rk.id in rj.dominates:
    #     [-Br] <- <+Ws> # a rule is blocked if it is dominated by an active rule s
                        pW[rj.id].addInVert(mB[rk.id])
    #     <+Br> <- [-Ws] # a rule is blocking only if all rules dominating it are inactive
                        mW[rj.id].addInVert(pB[rk.id])
            else:
                for rj in supporters[rk.consequent]:
                    if rk.id in rj.dominates:
                        pW[rj.id].addInVert(mB[rk.id])
                        mW[rj.id].addInVert(pB[rk.id])
# The defeasible extension: ambiguity blocking
    if not ambiguityPropagation:
        for r in theory:
    #     <+Ws> <- <+Ar> # a rule is active if it is applicable (there are no other edges towards <+Ws> anyway)
            pA[r.id].addInVert(pW[r.id])
    #     [-Ws] <- [-Ar] # a rule is inactive if it is inapplicable
            mA[r.id].addInVert(mW[r.id])
            for a in r.antecedent:
    #     [-Br] <- <-dq> # a rule is blocked if some of its antecedents are -d
                md[a].addInVert(mB[r.id])
    #     <+Br> <- [+dq] # a rule is blocking only if all its antecedents are +d
                pd[a].addInVert(pB[r.id])
    else:
# The defeasible extension: ambiguity propagation
        for l in literals:
    #     [+Sq] <- [+Dq] # q is supported if it is strictly provable
            pD[l].addInVert(pS[l])
    #     <-Sq> <- <-Dq> # q is unsupported only if it is not strictly provable
            mD[l].addInVert(mS[l])
        for r in theory:
    #     <+Wr> <- <+Ur> # a rule is active if it is supported (there are no other edges towars <+Wr> anyway)
            pU[r.id].addInVert(pW[r.id])
    #     [-Wr] <- [-Ur] # a rule is inactive if it is unsupported
            mU[r.id].addInVert(mW[r.id])
            if not (DEFEATER == r.operator):
    #     <+Pr> <- <-D-q> # a rule is plausible only if its consequent is not strictly opposed
                mD[-r.consequent].addInVert(pP[r.id])
    #     [-Pr] <- [+D-q] # a rule is implausible if its consequent is strictly opposed
                pD[-r.consequent].addInVert(mP[r.id])
    #     [+Sq] <- <+Pr> # q is supported if some rule with it as consequent is plausible
                pP[r.id].addInVert(pS[r.consequent])
    #     <-Sq> <- [-Pr] # q is unsupported if all rules with it as consequent are implausible
                mP[r.id].addInVert(mS[r.consequent])
                for rj in opponents[r.consequent]:
                    if r.id in rj.dominates:
    #     <+Pr> <- [-As] # a rule is plausible only if all rules s dominating it are inapplicable
                        mA[rj.id].addInVert(pP[r.id])
    #     [-Pr] <- <+As> # a rule is implausible if some rule s dominating it is applicable
                        pA[rj.id].addInVert(mP[r.id])
            for a in r.antecedent:
    #     <+Ur> <- [+Sq] # a rule is supported only if all its antecedents are +S
                pS[a].addInVert(pU[r.id])
    #     [-Ur] <- <-Sq> # a rule is unsupported if some of its antecedents are -S
                mS[a].addInVert(mU[r.id])
    #     [-Br] <- <-Sq> # a rule is blocked if some of its antecedents are -S
                mS[a].addInVert(mB[r.id])
    #     <+Br> <- [+Sq] # a rule is blocking only if all its antecedents are +S
                pS[a].addInVert(pB[r.id])
    #     <+Pr> <- [+Sq] # a rule is plausible only if all its antecedents are +S
                pS[a].addInVert(pP[r.id])
    #     [-Pr] <- <-Sq> # a rule is implausible if some of its antecedents are -S
                mS[a].addInVert(mP[r.id])
    defeasibleInference = [pD[l] for l in pD.keys()] + [mD[l] for l in mD.keys()] + [pd[l] for l in pd.keys()] + [md[l] for l in md.keys()] + [pA[r] for r in pA.keys()] + [mA[r] for r in mA.keys()] + [pT[r] for r in pT.keys()] + [mT[r] for r in mT.keys()] + [pB[r] for r in pB.keys()] + [mB[r] for r in mB.keys()] + [pW[r] for r in pW.keys()] + [mW[r] for r in mW.keys()]
    if ambiguityPropagation:
        defeasibleInference = defeasibleInference + [pS[l] for l in pS.keys()] + [mS[l] for l in mS.keys()] + [pP[r] for r in pP.keys()] + [mP[r] for r in mP.keys()] + [pU[r] for r in pU.keys()] + [mU[r] for r in mU.keys()]
    if wellFoundedness:
# The defeasible core
#     <fq> <- [rq] # a literal is founded only when it is foundation-reachable
#     <fq> <- [nq] # a literal is founded only when it is NOT proven -dq
#     [rq] <- <er> # a literal is foundation-reachable when it has an established rule
#                  # a literal is also foundation-reachable when it is proven +dq
#     <er> <- <fq> # a rule is established when all its antecedents are founded
#     [nq] <- empty # set inside the attractor if and only if -dq not proven
        erIns = {}
        fq = {l: OddVertex("f%d"%l) for l in literals} # <f>
        rq = {l: EvenVertex("r%d"%l) for l in literals} # [r]
        er = {r: OddVertex("e%d"%r) for r in rules} # <e>
        nq = {l: EvenVertex("n%d"%l) for l in literals} # [n]
        for l, v in fq.items():
            rq[l].addInVert(v)
            nq[l].addInVert(v)
        for r in theory:
            countVs = 0
            if not (DEFEATER == r.operator):
                for l in r.antecedent:
                    fq[l].addInVert(er[r.id])
                    countVs = countVs + 1
            erIns[r.id] = countVs
        for l, v in rq.items():
            for r in supporters[l]:
                er[r.id].addInVert(v)
        wellFoundedInference = [fq[l] for l in fq.keys()] + [rq[l] for l in rq.keys()] + [nq[l] for l in nq.keys()] + [er[l] for l in er.keys()]
    # Stage 3: run alternating fixpoints
    work = True
    first = True
    todoVerts = defeasibleInference
    #for v in defeasibleInference:
    #    if v.inverts:
    #        print(str(v))
    #        print("\t",[str(w) for w in v.inverts])
    while work:
        work = attractor(todoVerts)
        todoVerts = []
        if wellFoundedness:
            for v in wellFoundedInference:
                v.inAttractor = False
            for _, v in fq.items():
                v.safeOutverts = 2
            for k, v in er.items():
                v.safeOutverts = erIns[k]
                if 0 == v.safeOutverts:
                    v.inAttractor = True
            for l in literals:
                if pd[l].inAttractor:
                    rq[l].inAttractor = True
                if ambiguityPropagation:
                    if not mS[l].inAttractor:
                        nq[l].inAttractor = True
                else:
                    if not md[l].inAttractor:
                        nq[l].inAttractor = True
            attractor(wellFoundedInference)
            for l, v in fq.items():
                if (not v.inAttractor):
                    if (not md[l].inAttractor):
                        md[l].inAttractor = True
                        todoVerts.append(md[l])
                    if ambiguityPropagation and (not mS[l].inAttractor):
                        mS[l].inAttractor = True
                        todoVerts.append(mS[l])
        #print("todos", [str(x) for x in todoVerts])
        if [] == todoVerts:
            break
    # Stage 4: collect results
    for l in literals:
        if pd[l].inAttractor:
            retq.defeasiblyProvable.add(l) 
        elif md[l].inAttractor:
            retq.defeasiblyUnprovable.add(l)
        if ambiguityPropagation: 
            if pS[l].inAttractor:
                retq.supportable.add(l) 
            elif mS[l].inAttractor:
                retq.unsupportable.add(l) 
        else:
            if pd[l].inAttractor:
                retq.supportable.add(l) 
            elif md[l].inAttractor:
                retq.unsupportable.add(l) 
    return retq


class Variable:
    def __init__(self,name):
        self._name = name
    def getName(self):
        return str(self._name)
    def __str__(self):
        return str(self._name)
    def __gt__(self,b):
        return str(self) > str(b)
    def __ge__(self,b):
        return str(self) >= str(b)
    def __lt__(self,b):
        return str(self) < str(b)
    def __le__(self,b):
        return str(self) <= str(b)

class PFact:
    def __init__(self,p):
        self._p = p
        self._ps = {}
        self._po = {}
    def __str__(self):
        retq = ""
        for s, os in self._ps.items():
            for o in os:
                retq = retq + ("%s(%s, %s)" % (self._p,s,o))
        return retq
    def getTriples(self):
        retq = []
        for s, os in self._ps.items():
            for o in os:
                retq.append((self._p, s, o))
        return retq
    def getP(self):
        return str(self._p)
    def addFact(self, s, o, op):
        if (False == self.hasFact(s,o)):
            if s not in self._ps:
                self._ps[s] = set()
            if o not in self._po:
                self._po[o] = {}
            self._ps[s].add(o)
            if (s in self._po[o]) and (STRICT == self._po[o][s]):
                op = STRICT
            self._po[o][s] = op
    def delFact(self, s, o):
        if (False != self.hasFact(s, o)):
            self._ps[s].remove(o)
            self._po[o].pop(s)
    def getSFacts(self, s):
        if s in self._ps:
            return [(s, x) for x in self._ps[s]]
        return []
    def getSFactsSize(self, s):
        if s in self._ps:
            return len(self._ps[s])
        return 0
    def getOFacts(self, o):
        if o in self._po:
            return [(x, o) for x in self._po[o].keys()]
        return []
    def getOFactsSize(self, o):
        if o in self._po:
            return len(self._po[o])
        return 0
    def hasFact(self, s, o):
        if (s in self._ps) and (o in self._ps[s]):
            return self._po[o][s]
        return False
    def match(self, s, o, bdgs, usedFacts, newMask=None):
        def _updateUsedFacts(usedFacts, p, s, o):
            if usedFacts is not None:
                if p not in usedFacts:
                    usedFacts[p] = PFact(p)
                usedFacts[p].addFact(s, o, self._po[o][s])
        retq = []
        for bdg in bdgs:
            vs = s
            vo = o
            if isinstance(s, Variable):
                if s.getName() in bdg:
                    if bdg[s.getName()] in self._ps.keys():
                        vs = bdg[s.getName()]
                    else:
                        vs = None
            if isinstance(o, Variable):
                if o.getName() in bdg:
                    if bdg[o.getName()] in self._po.keys():
                        vo = bdg[o.getName()]
                    else:
                        vo = None
            if (None == vs) or (None == vo):
                continue
            if isinstance(vs, Variable):
                if isinstance(vo, Variable):
                    for fs, fos in self._ps.items():
                        for fo in fos:
                            if (None==newMask) or ((self._p, fs, fo) in newMask):
                                aux = bdg.copy()
                                aux.update({vs.getName(): fs, vo.getName(): fo})
                                retq.append(aux)
                                _updateUsedFacts(usedFacts, self._p, fs, fo)
                else:
                    for xso in self.getOFacts(vo):
                        if (None==newMask) or ((self._p, xso[0], vo) in newMask):
                            aux = bdg.copy()
                            aux[vs.getName()] = xso[0]
                            retq.append(aux)
                            _updateUsedFacts(usedFacts, self._p, xso[0], vo)
            elif isinstance(vo, Variable):
                for xso in self.getSFacts(vs):
                    if (None==newMask) or ((self._p, vs, xso[1]) in newMask):
                        aux = bdg.copy()
                        aux[vo.getName()] = xso[1]
                        retq.append(aux)
                        _updateUsedFacts(usedFacts, self._p, vs, xso[1])
            elif (False != self.hasFact(vs, vo)):
                if (None==newMask) or ((self._p, vs, vo) in newMask):
                    retq.append(bdg)
                    _updateUsedFacts(usedFacts, self._p, vs, vo)
        return retq, usedFacts
class RuleTemplate:
    def __init__(self,label,antecedent,operator,consequent):
        self._id = label
        self._antecedent = {k: a for k, a in enumerate(antecedent)}
        self._operator = operator
        self._consequent = tuple(consequent)
        self.dominates = set()
        self._pset = {}
        self._varMap = {}
        for k,a in self._antecedent.items():
            if a[0] not in self._pset:
                self._pset[a[0]] = []
            self._pset[a[0]].append(k)
            for x in a[1:]:
                if isinstance(x, Variable):
                    if x.getName() not in self._varMap:
                        self._varMap[x.getName()] = set()
                    self._varMap[x.getName()].add(k)
    def __str__(self):
        aux = ""
        if None != self._id:
            aux = "%s: " % str(self._id)
        return aux + termSet2String(self._antecedent) + {STRICT: " -> ", DEFEASIBLE: " => ", DEFEATER: " ~> "}[self._operator] + term2String(self._consequent)
    def instantiate(self, bdg):
        def opposing(e,f):
            pe = e[0]
            pf = f[0]
            if (pe == '-'+pf) or ('-'+pe == pf):
                return (e[1:] == f[1:])
            return False
        antecedent = []
        consequent = []
        bdgTp = tuple(sorted([(k,v) for k,v in bdg.items()]))
        for x in self._consequent:
            if isinstance(x, Variable) and (x.getName() in bdg):
                x = bdg[x.getName()]
            consequent.append(x)
        for _,x in self._antecedent.items():
            aux = []
            for a in x:
                if isinstance(a, Variable) and (a.getName() in bdg):
                    a = bdg[a.getName()]
                aux.append(a)
            aux = tuple(aux)
            antecedent.append(aux)
        for e in antecedent:
            for f in antecedent:
                if opposing(e,f):
                    return None
        return (tuple(antecedent), self._operator, tuple(consequent), (self._id, tuple(consequent), tuple(self.dominates), bdgTp))
    def _toposortAntecedents(self, startK):
        retq = []
        toVisit = [startK]
        visited = set()
        while toVisit:
            cr = toVisit.pop()
            if not cr in visited:
                visited.add(cr)
                retq.append(self._antecedent[cr])
                a = self._antecedent[cr]
                for x in a[1:]:
                    if isinstance(x, Variable):
                        for k in self._varMap[x.getName()]:
                            if k not in visited:
                                toVisit.append(k)
        if len(visited) < len(self._antecedent):
            for k, a in self._antecedent.items():
                if k not in visited:
                    retq.append(self._antecedent[k])
        return retq[1:]
    def getInstantiations(self, relevantFacts, knowledgeFacts, usedBackgroundFacts, newMask=None):
        relevantFacts = {x.getP():x for k,x in relevantFacts.items() if x.getP() in self._pset.keys()}
        #knowledgeFacts = {x.getP():x for k,x in knowledgeFacts.items() if x.getP() in self._pset.keys()}
        retq = set()
        for rp,rf in relevantFacts.items():
            for pm in self._pset[rf.getP()]:
                bdgs, _ = rf.match(self._antecedent[pm][1], self._antecedent[pm][2], [{}], None, newMask=newMask)
                if bdgs:
                    resorder = self._toposortAntecedents(pm)
                    for rk in resorder:
                        bdgsR = []
                        bdgsK = []
                        if rk[0] in relevantFacts:
                            bdgsR, _ = relevantFacts[rk[0]].match(rk[1], rk[2], bdgs, None)
                        if rk[0] in knowledgeFacts:
                            bdgsK, usedBackgroundFacts = knowledgeFacts[rk[0]].match(rk[1], rk[2], bdgs, usedBackgroundFacts)
                        bdgs = bdgsR + bdgsK
                        if not bdgs:
                            break
                    for bdg in bdgs:
                        newInstantiation = self.instantiate(bdg)
                        if newInstantiation:
                            retq.add(newInstantiation)
        return retq, usedBackgroundFacts
class TheoryTemplate:
    def __init__(self):
        self._rules = {}
        self._predTriggerMap = {}
    def __str__(self):
        retq = ""
        for srule in sorted(self._rules.keys()):
            retq = retq + srule + "\n"
        return retq[:-1]
    def addRules(self, rules):
        for rule in rules:
            srule = str(rule)
            if srule not in self._rules:
                self._rules[srule] = rule
                for predTrigger in rule._pset.keys():
                    if predTrigger not in self._predTriggerMap:
                        self._predTriggerMap[predTrigger] = set()
                    self._predTriggerMap[predTrigger].add(srule)
    def potentiallyTriggered(self, predicates):
        retq = set()
        for predicate in predicates:
            if predicate in self._predTriggerMap:
                retq = retq.union(self._predTriggerMap[predicate])
        return [self._rules[x] for x in retq]

def flip(consequent):
    retq = [c for c in consequent]
    if "-" == retq[0][0]:
        retq[0] = retq[0][1:]
    else:
        retq[0] = "-" + retq[0]
    return tuple(retq)
def netDominates(a, b):
    provA, consequentA, subsA, _ = a
    provB, consequentB, _, _ = b
    return (provB in subsA) and (consequentA == flip(consequentB))
def str2idxTheory(rules):
    s2i = {}
    i2s = {}
    theory = []
    r2w = {}
    i2r = {}
    supporters = {}
    for r in rules:
        antecedent, operator, consequent, netDom = r
        if consequent not in s2i:
            idx = len(s2i)+1
            if "-" == consequent[0][0]:
                idx = -idx
            i2s[idx] = consequent
            i2s[-idx] = flip(consequent)
            s2i[i2s[idx]] = idx
            s2i[i2s[-idx]] = -idx
        for a in antecedent:
            if a not in s2i:
                idx = len(s2i)+1
                if "-" == a[0][0]:
                    idx = -idx
                i2s[idx] = a
                i2s[-idx] = flip(a)
                s2i[i2s[idx]] = idx
                s2i[i2s[-idx]] = -idx
        aux = Rule()
        aux.id = len(theory)
        r2w[aux.id] = netDom
        aux.operator = operator
        aux.consequent = s2i[consequent]
        aux.antecedent = set([s2i[x] for x in antecedent])
        i2r[aux.id] = aux
        sc = s2i[consequent]
        if DEFEATER == operator:
            sc = -sc
        if sc not in supporters:
            supporters[sc] = []
        supporters[sc].append(aux.id)
        theory.append(aux)
    for sc in supporters.keys():
        if (0 > sc) or (-sc not in supporters):
            continue
        for rs in supporters[sc]:
            for ro in supporters[-sc]:
                if (None != r2w[rs]) and (None != r2w[ro]):
                    if netDominates(r2w[ro], r2w[rs]):
                        i2r[ro].dominates.add(rs)
                    elif netDominates(r2w[rs], r2w[ro]):
                        i2r[rs].dominates.add(ro)
    return s2i, i2s, theory

def idx2strConclusions(conclusions, i2s):
    retq = Conclusions()
    for cset, rset in [(conclusions.strictlyProvable, retq.strictlyProvable), (conclusions.strictlyUnprovable, retq.strictlyUnprovable), (conclusions.defeasiblyProvable, retq.defeasiblyProvable), (conclusions.defeasiblyUnprovable, retq.defeasiblyUnprovable), (conclusions.supportable, retq.supportable), (conclusions.unsupportable, retq.unsupportable), (conclusions.contradictions, retq.contradictions)]:
        for idx in cset:
            rset.add(i2s[idx])
    return retq

def readTerm(s):
    lp = s.find('(')
    rp = s.find(')')
    if (-1 in [lp, rp]) and (lp != rp):
        raise ValueError("Term contains mismatched parantheses: %s" % s)
    if -1 == lp:
        return (s, '', '')
    p = s[:lp].strip()
    args = [x.strip() for x in s[lp+1:rp].split(',')]
    if 2 < len(args):
        raise ValueError("Term contains a predicate with more than two arguments: %s" % s)
    if 0 == len(args):
        args.append('')
    if 1 == len(args):
        args.append('')
    if (0 < len(args[0])) and ('?' == args[0][0]):
        args[0] = Variable(args[0])
    if (0 < len(args[1])) and ('?' == args[1][0]):
        args[1] = Variable(args[1])
    return (p, args[0], args[1])
def splitTerms(s):
    retq = []
    aux = ''
    inPar = False
    for c in s:
        if (',' == c) and (not inPar):
            retq.append(aux)
            aux = ''
            continue
        if '(' == c:
            if inPar:
                raise ValueError("Antecedent contains complex arguments: %s" % s)
            inPar = True
        elif ')' == c:
            inPar = False
        aux = aux + c
    retq.append(aux)
    return retq
def term2String(term):
    retq = str(term[0])
    if (1 < len(term)) and (('' != str(term[1])) or ((2 < len(term)) and ('' != str(term[2])))):
        retq = retq + "(" + str(term[1])
        if (2 < len(term)) and ('' != str(term[2])):
            retq = retq + ", " + str(term[2])
        retq = retq + ")"
    return retq
def termSet2String(terms):
    if isinstance(terms,dict):
        terms = sorted([v for _,v in terms.items()])
    elif isinstance(terms,tuple) or isinstance(terms, list):
        terms = sorted(list(terms))
    retq = ''
    for k, t in enumerate(terms):
        retq = retq + term2String(t)
        if len(terms) > k + 1:
            retq = retq + ', '
    return retq
def getStrTheory(groundRules):
    retq = ""
    supporters = {}
    for k,r in enumerate(groundRules):
        antecedent, operator, consequent, netDom = r
        aux = ("%d: " % k) + termSet2String(antecedent) + (" %s> "%({STRICT: "-", DEFEASIBLE: "=", DEFEATER: "~"}[operator])) + term2String(consequent)
        if DEFEATER == operator:
            consequent = flip(consequent)
        if consequent not in supporters:
            supporters[consequent] = set()
        supporters[consequent].add((k, netDom))
        retq = retq + aux + "\n"
    for sc in supporters.keys():
        if ("-" == sc[0][0]) or (flip(sc) not in supporters):
            continue
        fsc = flip(sc)
        for rs in supporters[sc]:
            ks, ndS = rs
            for ro in supporters[fsc]:
                ko, ndO = ro
                if (None != ndS) and (None != ndO):
                    if netDominates(ndO, ndS):
                        retq = retq + ("%d > %d\n" % (ko, ks))
                    elif netDominates(ndS, ndO):
                        retq = retq + ("%d > %d\n" % (ks, ko))
    return retq
def loadDFLRules(inFile, rules=None):
    def newLabel(d):
        retq = len(d)
        k = 0
        while retq in d:
            retq = retq + 2*k+1
            k = k + 1
        return retq
    if None == rules:
        rules = TheoryTemplate()
    # Kinds of lines:
    # label: a,b,... => c
    # labelA > labelB
    retq = {}
    lines = [y for y in [x.strip() for x in open(inFile).read().splitlines()] if y and ('#'!=y[0])]
    drel = set()
    for l in lines:
        if l.startswith("!INCLUDE "):
            includedName = (l[len("!INCLUDE "):]).strip()
            loadDFLRules(includedName, rules)
        lhs, rhs = l.split('>')
        opT = lhs[-1]
        if opT in ["-", "=", "~"]:
            operator = {"-": STRICT, "=": DEFEASIBLE, "~": DEFEATER}[opT]
            labelled_antecedent, consequent = lhs, rhs
            labelled_antecedent = labelled_antecedent[:-1]
            label = newLabel(retq)
            antecedent = labelled_antecedent
            if -1 != labelled_antecedent.find(':'):
                label, antecedent = labelled_antecedent.split(':')
                label = label.strip()
            consequent = readTerm(consequent)
            antecedent = [readTerm(x) for x in splitTerms(antecedent.strip())]
            retq[label] = RuleTemplate(label, antecedent, operator, consequent)
        else:
            drel.add((lhs.strip(), rhs.strip()))
    for d in drel:
        dom, sub = d
        if dom in retq:
            retq[dom].dominates.add(sub)
    rules.addRules([retq[k] for k in retq.keys()])
    return rules
def loadDFLFacts(inFile, facts=None):
    if None == facts:
        facts = {}
    lines = [y for y in [x.strip() for x in open(inFile).read().splitlines()] if y and ('#'!=y[0])]
    for l in lines:
        decorator, term = l.split('>')
        term = readTerm(term)
        operator = {"-": STRICT, "=": DEFEASIBLE}[decorator[-1]]
        if term[0] not in facts:
            facts[term[0]] = PFact(term[0])
        facts[term[0]].addFact(term[1], term[2], operator)
    return facts

def buildTheory(rules, relevantFacts, knowledgeBase, debugTheory=False):
    theoryStr = ''
    groundRules = set()
    relevantFacts = relevantFacts.copy()
    usedBackgroundFacts = {}
    newPredicates = set()
    for rp, rf in relevantFacts.items():
        newPredicates.add(rp)
        for triple in rf.getTriples():
            operator = rf.hasFact(triple[1], triple[2])
            groundRules.add(((), operator, triple, None))
    newMask = None
    visitedMask = set()
    newAdditions = True
    while newAdditions:
        newAdditions = False
        auxMask = set()
        auxMaskNew = set()
        crPredicates = newPredicates
        newPredicates = set()
        for r in rules.potentiallyTriggered(crPredicates):
            instantiations, usedBackgroundFacts = r.getInstantiations(relevantFacts, knowledgeBase, usedBackgroundFacts, newMask=newMask)
            if instantiations:
                newAdditions = True
            for i in instantiations:
                antecedent, operator, consequent, netDom = i
                if netDom in visitedMask:
                #if consequent in visitedMask:
                    continue
                newPredicates.add(consequent[0])
                auxMask.add(netDom)
                #auxMask.add(consequent)
                auxMaskNew.add(consequent)
                if consequent[0] not in relevantFacts:
                    relevantFacts[consequent[0]] = PFact(consequent[0])
                relevantFacts[consequent[0]].addFact(consequent[1],consequent[2],DEFEASIBLE)
                groundRules.add(i)
        newMask = auxMaskNew
        visitedMask = visitedMask.union(auxMask)
        #visitedMask = visitedMask.union(auxMask)
    for rp, rf in usedBackgroundFacts.items():
        for triple in rf.getTriples():
            operator = rf.hasFact(triple[1], triple[2])
            groundRules.add(((), operator, triple, None))
    s2i, i2s, theory = str2idxTheory(groundRules)
    if debugTheory:
        theoryStr = getStrTheory(groundRules)
    return theory, s2i, i2s, theoryStr

