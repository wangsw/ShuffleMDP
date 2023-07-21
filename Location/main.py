__author__ = 'westbrick'

import numpy as np
import scipy as sp
import numpy.random as r
import distance
import math
from sympy import S, Symbol, nsolve, log, exp, sqrt, Piecewise
import mpmath


import simulator
import tester


def multiuniform(ranges, sample, norm=distance.l2norm, normalize=True):
    points = np.zeros((sample, len(ranges)))
    for j in range(0, len(ranges)):
        points[:, j] = np.random.choice(np.arange(ranges[j]), sample, False)
    distances = distance.todistances(points, norm)
    maxdis = np.max(distances)
    #for i in range(0, sample):
    #    distances[i, i] = maxdis
    mindis = np.min(distances[np.nonzero(distances)])
    if normalize == True:
        points /= mindis
    return points


def msolve(formula, v, init, T=50):
    # my solver via binary search with extension, find the largest non-negative v that formuala <= 0
    vl = 0.0
    vr = 2*init
    for t in range(T):
        vt = (vl+vr)/2
        #print("vl, vr", vl, vr)
        if (not formula.subs(v, vt).is_real) or formula.subs(v, vt) > 0:
            vr = vt
        elif formula.subs(v, vr).is_real and formula.subs(v, vr) < 0:
            vr = 2*vr
        elif formula.subs(v, vt) < 0:
            vl = vt
        else:
            return vt
    return vl



def amplification(ds, m, n, delta):
    # use tighter amplification bound and return local metric
    # m is the largest distance in the local
    dmin = np.min(ds[np.nonzero(ds)])
    dmax = np.max(ds)
    v = Symbol("v", real=True)
    Omega = Symbol("Omega", real=True)
    Omegav = n/exp(v)-sqrt(2*n/exp(v)*log(2/delta))
    FvOmega = log(1+(exp(v)-1)/(exp(v)+1)*\
                (2*sqrt(Omega/2*log(4/delta))+1)/(Omega/2-sqrt(Omega/2*log(4/delta))))
    low = msolve(FvOmega.subs(Omega, Omegav)-dmin, v, dmin) #.evalf(chop=True)
    high2 = msolve(-Omegav/2+sqrt(Omegav/2*log(4/delta)), v, dmax*low/dmin) #.evalf(chop=True)
    #print("low, high2", low, high2)
    high = msolve(FvOmega.subs(Omega, Omegav)-dmax, v, high2/2) #.evalf(chop=True)
    #print("high, high2", high, high2, (Omegav/2-sqrt(Omegav/2*log(4/delta))).subs(v, high).evalf())
    realhigh = min(high, high2)
    high = max(dmax, realhigh)

    #print("feasible range of m", low, high)
    if m is None:
        # just return low, high
        return float(low), float(high), float(realhigh)
    elif m < low or m > high:
        print("warning: m is not in feasible range", m, (low, high))
        #print("m is not in feasible range, changed to", (low+high)/2)
        #m = (low+high)/2

    # local metric
    cds = np.copy(ds)
    nds = cds.shape
    D = np.full_like(ds, -1.0, dtype=float)
    prevmax = m

    while np.max(cds) > 0.0:
        one_exp = 0.0
        two_exp = 0.0
        currentdist = np.max(cds)
        for a in range(nds[0]):
            for b in range(a + 1, nds[0]):
                if cds[a][b] == currentdist:
                    for c in range(nds[0]):
                        if D[a][c] >= 0 and D[b][c] >= 0:
                            two_exp = max(two_exp, np.exp(D[a][c]) + np.exp(D[b][c]))
                        if D[a][c] >= 0 and D[b][c] < 0:
                            one_exp = max(one_exp, np.exp(D[a][c]))
                        if D[a][c] < 0 and D[b][c] >= 0:
                            one_exp = max(one_exp, np.exp(D[b][c]))
        two_exp = max(one_exp, two_exp)
        maxexpr = Piecewise((2*exp(v), (2*exp(v)>=one_exp+exp(v)) & (2*exp(v)>=two_exp)),
                            (one_exp+exp(v), (one_exp+exp(v)>=2*exp(v)) & (one_exp+exp(v)>=two_exp)),
                            (two_exp, True)
                            )
        #print("maxexpr", currentdist, one_exp, two_exp, maxexpr.subs(v, currentdist))
        Omegav = 2*n/maxexpr-sqrt(4*n/maxexpr*log(2/delta))
        #print("nsolve", currentdist, prevmax, maxexpr, FvOmega.subs(Omega, Omegav))
        #print("amplification", currentdist, m, one_exp, two_exp, (Omegav/2-sqrt(Omegav/2*log(4/delta))).subs(v, m).evalf(), ((Omegav/2-sqrt(Omegav/2*log(4/delta))).subs(v, currentdist)).evalf(), FvOmega.subs(Omega, Omegav).subs(v, currentdist).evalf(), currentdist)
        if (not (Omegav/2-sqrt(Omegav/2*log(4/delta))).subs(v, currentdist).is_real) or (Omegav/2-sqrt(Omegav/2*log(4/delta))).subs(v, currentdist) < 0 or FvOmega.subs(Omega, Omegav).subs(v, currentdist) > currentdist:
            dist = currentdist
        elif FvOmega.subs(Omega, Omegav).subs(v, prevmax).is_real and FvOmega.subs(Omega, Omegav).subs(v, prevmax) <= currentdist:
            dist = prevmax
        else:
            dist = msolve(FvOmega.subs(Omega, Omegav) - currentdist, v, (prevmax+currentdist)/2) #.evalf(chop=True)

        prevmax = min(prevmax, dist)
        D[np.where(cds == currentdist)] = prevmax
        # marked as processed
        cds[np.where(cds == currentdist)] = 0
    np.fill_diagonal(D, 0.0)

    # shorted path
    D = sp.sparse.csgraph.shortest_path(D, method='auto', directed=False, return_predecessors=False)
    return D


def amplificationSimpler(ds, m, n, delta):
    # use tighter amplification bound and return local metric
    # m is the largest distance in the local
    dmin = np.min(ds[np.nonzero(ds)])
    dmax = np.max(ds)
    v = Symbol("v", real=True)
    Omega = Symbol("Omega", real=True)
    Omegav = n/exp(v)-sqrt(2*n/exp(v)*log(2/delta))
    FvOmega = log(1+(exp(v)-1)/(exp(v)+1)*\
                (2*sqrt(Omega/2*log(4/delta))+1)/(Omega/2-sqrt(Omega/2*log(4/delta))))
    low = msolve(FvOmega.subs(Omega, Omegav)-dmin, v, dmin) #.evalf(chop=True)
    high2 = msolve(-Omegav/2+sqrt(Omegav/2*log(4/delta)), v, dmax*low/dmin) #.evalf(chop=True)
    #print("low, high2", low, high2)
    high = msolve(FvOmega.subs(Omega, Omegav)-dmax, v, high2/2) #.evalf(chop=True)
    #print("high, high2", high, high2, (Omegav/2-sqrt(Omegav/2*log(4/delta))).subs(v, high).evalf())
    realhigh = min(high, high2)
    high = max(dmax, realhigh)

    #print("feasible range of m", low, high)
    if m is None:
        # just return low, high
        return float(low), float(high), float(realhigh)
    elif m < low or m > high:
        print("warning: m is not in feasible range", m, (low, high))
        #print("m is not in feasible range, changed to", (low+high)/2)
        #m = (low+high)/2

    # local metric
    cds = np.copy(ds)
    nds = cds.shape
    D = np.full_like(ds, -1.0, dtype=float)
    prevmax = m

    while np.max(cds) > 0.0:
        one_exp = 0.0
        two_exp = 0.0
        currentdist = np.max(cds)
        # for a in range(nds[0]):
        #     for b in range(a + 1, nds[0]):
        #         if cds[a][b] == currentdist:
        #             for c in range(nds[0]):
        #                 if D[a][c] >= 0 and D[b][c] >= 0:
        #                     two_exp = max(two_exp, np.exp(D[a][c]) + np.exp(D[b][c]))
        #                 if D[a][c] >= 0 and D[b][c] < 0:
        #                     one_exp = max(one_exp, np.exp(D[a][c]))
        #                 if D[a][c] < 0 and D[b][c] >= 0:
        #                     one_exp = max(one_exp, np.exp(D[b][c]))
        # two_exp = max(one_exp, two_exp)
        maxexpr = 2*exp(m)
        #print("maxexpr", currentdist, one_exp, two_exp, maxexpr.subs(v, currentdist))
        Omegav = 2*n/maxexpr-sqrt(4*n/maxexpr*log(2/delta))
        #print("nsolve", currentdist, prevmax, maxexpr, FvOmega.subs(Omega, Omegav))
        #print("amplification", currentdist, m, one_exp, two_exp, (Omegav/2-sqrt(Omegav/2*log(4/delta))).subs(v, m).evalf(), ((Omegav/2-sqrt(Omegav/2*log(4/delta))).subs(v, currentdist)).evalf(), FvOmega.subs(Omega, Omegav).subs(v, currentdist).evalf(), currentdist)
        if (not (Omegav/2-sqrt(Omegav/2*log(4/delta))).subs(v, currentdist).is_real) or (Omegav/2-sqrt(Omegav/2*log(4/delta))).subs(v, currentdist) < 0 or FvOmega.subs(Omega, Omegav).subs(v, currentdist) > currentdist:
            dist = currentdist
        elif FvOmega.subs(Omega, Omegav).subs(v, prevmax).is_real and FvOmega.subs(Omega, Omegav).subs(v, prevmax) <= currentdist:
            dist = prevmax
        else:
            dist = msolve(FvOmega.subs(Omega, Omegav) - currentdist, v, (prevmax+currentdist)/2) #.evalf(chop=True)

        prevmax = min(prevmax, dist)
        D[np.where(cds == currentdist)] = prevmax
        # marked as processed
        cds[np.where(cds == currentdist)] = 0
    np.fill_diagonal(D, 0.0)

    # shorted path
    D = sp.sparse.csgraph.shortest_path(D, method='auto', directed=False, return_predecessors=False)
    return D


# def amplificationSimpler(ds, m, n, delta):
#     # use tighter amplification bound and return local metric
#     # m is the largest distance in the local
#     dmin = np.min(ds[np.nonzero(ds)])
#     dmax = np.max(ds)
#     v = Symbol("v", real=True)
#     u = Symbol("u", real=True) # max_{c} exp(a,c)+exp(b,c)
#     Fvu = log(1+8*(exp(v)-1)/(exp(v)+1)*(sqrt(0.5*u*log(4/delta)/n)+0.5*u/n))
#     low = nsolve(Fvu.subs(u, 2*exp(v))-dmin, dmin).evalf(chop=True)
#     high = nsolve(Fvu.subs(u, 2*exp(v))-dmax, (low+dmax)/2).evalf(chop=True)
#     if high is complex:
#         high = high.real
#     realhigh = min(high, nsolve(n-16*log(2/delta)*exp(v), np.log(n)).evalf(chop=True))
#     high = max(dmax, realhigh)
#
#     #print("feasible range of m", low, high)
#     if m is None:
#         # just return low, high
#         return float(low), float(high), float(realhigh)
#     elif m < low or m > high:
#         print("warning: m is not in feasible range", m, (low, high))
#         #print("m is not in feasible range, changed to", (low+high)/2)
#         #m = (low+high)/2
#
#     # local metric
#     cds = np.copy(ds)
#     nds = cds.shape
#     D = np.full_like(ds, -1.0, dtype=float)
#     prevmax = m
#
#     while np.max(cds) > 0.0:
#         one_exp = 0.0
#         two_exp = 0.0
#         currentdist = np.max(cds)
#         for a in range(nds[0]):
#             for b in range(a + 1, nds[0]):
#                 if cds[a][b] == currentdist:
#                     for c in range(nds[0]):
#                         if D[a][c] >= 0 and D[b][c] >= 0:
#                             two_exp = max(two_exp, np.exp(D[a][c]) + np.exp(D[b][c]))
#                         if D[a][c] >= 0 and D[b][c] < 0:
#                             one_exp = max(one_exp, np.exp(D[a][c]))
#                         if D[a][c] < 0 and D[b][c] >= 0:
#                             one_exp = max(one_exp, np.exp(D[b][c]))
#         two_exp = max(one_exp, two_exp)
#         maxexpr = Piecewise((2*exp(v), (2*exp(v)>=one_exp+exp(v)) & (2*exp(v)>=two_exp)),
#                             (one_exp+exp(v), (one_exp+exp(v)>=2*exp(v)) & (one_exp+exp(v)>=two_exp)),
#                             (two_exp, True)
#                             )
#         #print("maxexpr", one_exp, two_exp, maxexpr)
#         #print("amplificationsimpler", currentdist, one_exp, two_exp, Fvu.subs(u, maxexpr).subs(v, currentdist).evalf(), currentdist)
#         if (n-6*log(2/delta)*maxexpr).subs(v, currentdist) < 0 or Fvu.subs(u, maxexpr).subs(v, currentdist) > currentdist:
#             dist = currentdist
#         elif Fvu.subs(u, maxexpr).subs(v, prevmax) < currentdist:
#             dist = prevmax
#         else:
#             dist = msolve(Fvu.subs(u, maxexpr) - currentdist,  v, (prevmax+currentdist)/2) #.evalf(chop=True)
#
#         prevmax = min(prevmax, dist)
#         D[np.where(cds == currentdist)] = prevmax
#         # marked as processed
#         cds[np.where(cds == currentdist)] = 0
#     np.fill_diagonal(D, 0.0)
#
#     # shorted path
#     D = sp.sparse.csgraph.shortest_path(D, method='auto', directed=False, return_predecessors=False)
#     return D


def amplificationSimplest(ds, m, n, delta):
    # use simpler amplfication bound and return local metric
    dmin = np.min(ds[np.nonzero(ds)])
    dmax = np.max(ds)
    v = Symbol("v", real=True)
    u = Symbol("u", real=True)
    Gvu = log(1+8*(exp(v)-1)/(exp(v)+1)*(sqrt(exp(u)*log(4/delta)/n)+exp(u)/n))

    #print(dmin, dmax, Gvu, Gvu.subs(u, v))

    low = nsolve(Gvu.subs(u, v)-dmin, dmin).evalf(chop=True)
    high = nsolve(Gvu.subs(u, v)-dmax, (low+dmax)/2).evalf(chop=True)
    realhigh = min(high, nsolve(n-16*log(4/delta)*exp(v), np.log(n)).evalf(chop=True))
    high = max(realhigh, dmax)
    #print("feasible range of m", low, high)
    if m is None:
        # just return low, high
        return float(low), float(high), float(realhigh)
    elif m < low or m > high:
        print("warning: m is not in feasible range", m, (low, high))
        #print("m is not in feasible range, changed to", (low+high)/2)
        #m = (low+high)/2


    # local metric
    cds = np.copy(ds)
    D = np.zeros_like(ds, dtype=float)

    prevmax = m

    while np.max(cds) > 0.0:
        currentdist = np.max(cds)
        if (n-16*log(4/delta)*exp(m)).subs(v, currentdist) < 0 or Gvu.subs([(v, currentdist), (u, m)]) > currentdist:
            dist = currentdist
        elif Gvu.subs([(v, prevmax), (u, m)]) <= currentdist:
            dist = prevmax
        else:
            #print("nsolve", currentdist, m, prevmax, cds)
            dist = msolve(Gvu.subs(u, m)-currentdist, v, (prevmax+currentdist)/2) #.evalf(chop=True)

        prevmax = min(prevmax, dist)
        D[np.where(cds == currentdist)] = prevmax
        # marked as processed
        cds[np.where(cds == currentdist)] = 0
    # shorted path
    D = sp.sparse.csgraph.shortest_path(D, method='auto', directed=False, return_predecessors=False)
    #print("shorted path", D)
    return D


#n = 134237 # Gowalla-SF-peninsula, https://www.petsymposium.org/2017/papers/issue4/paper67-2017-4-source.pdf
n = 1
shuffle_n = 134237 #20000# 134237 # the number of users for shuffling may differ from for aggregation, e.g., in Location-based systems
debias = False
# ds = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
# ds = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#width = np.arange(12)
#height = np.arange(28)
width = np.arange(3)
height = np.arange(7)
nds = np.array([len(width)*len(height)])
#nds = [6]
# wss = [[40]*8+[1 for i in range(8, d)] for d in nds]
#wss = [[np.exp(-2.0*i) for i in range(0, d)] for d in nds]
wss = np.array([[1.0 for i in range(0, d)] for d in nds])
#nds = [8, 16, 32]
# ds = [12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
# ds = [4]
# ds = [40, 50]
# eps = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0]
# eps = [0.001, 0.01, 0.05, 0.1, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0,  5.0]
#eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
#eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.4, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
eps = [0.2]
#eps = [0.3, 0.5]
#eps = [0.7, 0.9]
#eps = [1.1, 1.3, 1.7]
#eps = [2]
#eps = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 2.0, 2.3, 2.6, 3.0]
#eps = [100]
# eps = [1.0]
# eps = [0.001, 0.01, 1.0]
#repeat = 50000
#repeat = 2000
repeat = 8000

#pss = [np.arange(nds[i]) for i in range(0, len(nds))]
pss = [np.transpose([np.repeat(width, len(height)), np.tile(height, len(width))]) for i in range(len(nds))]
#pss =

#pss = [np.sort(np.random.choice(np.arange(nds[i]*2), nds[i], False)) for i in range(0, len(nds))]
#pss = [multiuniform((nds[i]*4, nds[i]*4), nds[i]) for i in range(0, len(nds))]
dss = [eps[0]*distance.todistances(pss[i], distance.l2norm) for i in range(len(nds))]
histogram = None

print("pss", pss)
print("dss", dss)

low = None
high = None

delta = 1e-7 # 1/shuffle_n
#amplificationFunc = amplificationSimplest
#"""
### begin amplification
lows = [0.0]*3
highs = [0.0]*3
realhighs = [0.0]*3
lows[0], highs[0], realhighs[0] = amplificationSimplest(dss[0], None, shuffle_n, delta)
lows[1], highs[1], realhighs[1] = amplificationSimpler(dss[0], None, shuffle_n, delta)
lows[2], highs[2], realhighs[2] = amplification(dss[0], None, shuffle_n, delta)
low = np.min(lows)
high = np.max(highs)
realhigh = np.max(realhighs)

print("amplified diameters:", (low, high, realhigh), (lows[0], highs[0], realhighs[0]), (lows[1], highs[1], realhighs[1]), (lows[2], highs[2], realhighs[2]))

#print("amplifiedSimplest distance test", amplificationSimplest(dss[0], (low+realhigh)/2, shuffle_n, 1.0/shuffle_n))
#print("amplifiedSimpler distance test", amplificationSimpler(dss[0], (low+realhigh)/2, shuffle_n, 1.0/shuffle_n))
#print("amplified distance test", amplification(dss[0], (low+realhigh)/2, shuffle_n, 1.0/shuffle_n))


diameters = list(np.arange(low, high, (high-low)/20.0))
# add critical diameters
for v in lows+highs+realhighs:
    if v not in diameters:
        diameters.append(v)
diameters = np.array(sorted(diameters))


pss = np.repeat(pss, 1+3*len(diameters), axis=0)
wss = np.repeat(wss, 1+3*len(diameters), axis=0)
dss = dss\
      +[amplificationSimplest(dss[0], diameters[i], shuffle_n, delta) for i in range(len(diameters))]\
      +[amplificationSimpler(dss[0], diameters[i], shuffle_n, delta) for i in range(len(diameters))]\
      +[amplification(dss[0], diameters[i], shuffle_n, delta) for i in range(len(diameters))]
nds = np.repeat(nds, 1+3*len(diameters), axis=0)
### end amplification
#"""
# shorted path algorithm

"""
pss = [np.array([[-41.24375, -157.5], [-41.24375, -112.5], [-41.24375, -67.5], [-41.24375, -22.5], [-41.24375, 22.5], [-41.24375, 67.5], [-41.24375, 112.5], [-41.24375, 157.5], [-23.731250000000003, -157.5], [-23.731250000000003, -112.5], [-23.731250000000003, -67.5], [-23.731250000000003, -22.5], [-23.731250000000003, 22.5], [-23.731250000000003, 67.5], [-23.731250000000003, 112.5], [-23.731250000000003, 157.5], [-6.21875, -157.5], [-6.21875, -112.5], [-6.21875, -67.5], [-6.21875, -22.5], [-6.21875, 22.5], [-6.21875, 67.5], [-6.21875, 112.5], [-6.21875, 157.5], [11.293749999999996, -157.5], [11.293749999999996, -112.5], [11.293749999999996, -67.5], [11.293749999999996, -22.5], [11.293749999999996, 22.5], [11.293749999999996, 67.5], [11.293749999999996, 112.5], [11.293749999999996, 157.5], [28.80624999999999, -157.5], [28.80624999999999, -112.5], [28.80624999999999, -67.5], [28.80624999999999, -22.5], [28.80624999999999, 22.5], [28.80624999999999, 67.5], [28.80624999999999, 112.5], [28.80624999999999, 157.5], [46.318749999999994, -157.5], [46.318749999999994, -112.5], [46.318749999999994, -67.5], [46.318749999999994, -22.5], [46.318749999999994, 22.5], [46.318749999999994, 67.5], [46.318749999999994, 112.5], [46.318749999999994, 157.5], [63.83125, -157.5], [63.83125, -112.5], [63.83125, -67.5], [63.83125, -22.5], [63.83125, 22.5], [63.83125, 67.5], [63.83125, 112.5], [63.83125, 157.5], [81.34375, -157.5], [81.34375, -112.5], [81.34375, -67.5], [81.34375, -22.5], [81.34375, 22.5], [81.34375, 67.5], [81.34375, 112.5], [81.34375, 157.5]]
) for i in range(0, len(nds))]

dss = [distance.todistances(pss[i], distance.l2norm) for i in range(0, len(nds))]


for i in range(0, len(nds)):
    maxdis = np.max(dss[i])
    mindis = np.min(dss[i][np.nonzero(dss[i])])
    pss[i] /= mindis

#print([np.min(dss[i]) for i in range(0, len(nds))])

dss = [eps[i]*distance.todistances(pss[i], distance.l2norm) for i in range(0, len(nds))]




histogram = np.array([0, 2, 0, 410, 8501, 0, 669, 0, 0, 2, 0, 1058, 246535, 117679, 75, 1, 470, 1195, 145, 6528, 79963, 143589, 1, 0, 0, 694, 35, 6, 1382, 43842, 2263, 0, 687, 1053, 59, 50, 7001, 118479, 161194, 10, 0, 78, 1, 657, 8185, 58, 45, 0, 22, 456, 6046, 15553, 6474, 1249, 2, 0, 7654, 2200, 3, 246, 7338, 155, 0, 0])
histogram = np.floor(histogram/40.0)
n = int(math.floor(np.sum(histogram)))
"""

print('n=', n, 'shullfe_n', shuffle_n, 'debias', debias, 'repeat', repeat)
print('eps=',eps)
print('nds=', nds)
print("diamters", len(diameters), diameters)
#mechanisms = ['MRR', 'BRR', 'GBFMM', 'HBFMM', 'KSS', 'EM', 'EKSE', 'OKSE', 'ECSE', 'OCSE']
#mechanisms = ['EM', 'EKSE', 'OKSE', 'ECSE', 'OCSE']
#mechanisms = ['EM', 'KSS', 'HBFMM',  'EKSE', 'OKSE', 'ECSE', 'OCSE']
#mechanisms = ['EM', 'KSS', 'HBFMM',  'EKSE', 'OKSE']
#mechanisms = ['KSS', 'EM', 'HBFMM', 'EKSE', 'OKSE']
##mechanisms = ['EM', 'CEM', 'KSS', 'OKSE', 'OCSE'] #, 'ECSE', 'OCSE']

#mechanisms = ['EM', 'CEM', 'KSS', 'OKSE', 'OCSE']
mechanisms = ['EM', 'CEM']



simulation = simulator.Simulator()
simulation.init(n, nds, wss, pss, dss, eps, repeat, mechanisms, histogram, shuffle_n, debias, diameters, [lows, highs, realhighs])
simulation.simulate()
#simulation.write('gowalla_'+str(len(width))+'_'+str(len(height))+'_'+str(eps[0])+'n_'+str(n)+'_sn_'+str(shuffle_n)+'_debias'+str(debias)+'.json')




