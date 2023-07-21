__author__ = 'westbrick'
# util functions

import math
import numpy as np
import scipy as sp
import numpy.random as r
import mrr, brr, ksubset, hbfmm, gbfmm, setexponential, em, bpm

def binarysearch(l, v):
    # search the corresponding scope holding v
    s = 0
    e = len(l)-1
    while v < l[math.floor((s+e)/2)] or v >= l[math.floor((s+e)/2) + 1]:
        if v < l[math.floor((s+e)/2)]:
            e = math.floor((s+e)/2)
        else:
            s = math.floor((s+e)/2)
    return math.floor((s+e)/2)


def reservoirsample(l, m):
    # sample m elements from list l
    samples = l[0:m]
    for i in range(m, len(l)):
        index = r.randint(0, i+1)
        if index < m:
            samples[index] = l[i]
    return samples


def recorder(hits, pub):
    # record pub to hits
    for s in pub:
        hits[s] += 1
    return hits


def histogramer(d, n, dist=None):
    # create a size n histogram according to distribution dist
    if dist is None:
        dist = [1/d]*d
    h = np.full((d,), 0, dtype=int)
    for i in range(0, d):
        h[i] = int(r.binomial(n, dist[i]))
    while np.sum(h) != n:
        diff = n-np.sum(h)
        ri = r.randint(0, d-1)
        if h[ri]+diff >= 0:
            h[ri] = h[ri]+diff
        else:
            h[ri] = 0
    #print(h)
    return h


def projector(od):
    # project od to probability simplex
    u = -np.sort(-od)
    #print("sorted:\t", u)
    sod = np.zeros(len(od))
    sod[0] = u[0]
    for i in range(1, len(od)):
        sod[i] = sod[i-1]+u[i]

    for i in range(0, len(od)):
        sod[i] = u[i]+(1.0-sod[i])/(i+1)

    p = 0
    for i in range(len(od)-1, -1, -1):
        if sod[i] > 0.0:
            p = i
            break

    q = sod[p]-u[p]

    x = np.zeros(len(od))
    for i in range(0, len(od)):
        x[i] = np.max([od[i]+q, 0.0])
    #print("projected:\t",x)
    return x




def distributor(n, histogram, mechanism, debias):
    # randomize items in the hitogram and return observed hits
    hits = np.full(len(histogram), 0, dtype=int)
    for i in range(0, len(histogram)):
        for c in range(0, histogram[i]):
            recorder(hits, mechanism.randomizer(i))
    return mechanism.decoder(hits, n, debias)

def circleUp(ds, ps):
    cds = np.copy(ds)
    dims = np.max(ps, axis=0, keepdims=False)+1

    rps = np.arange(len(ps)).reshape(dims)
    reference = rps[tuple(((dims-1)//2).reshape((len(dims),1)))][0]
    #print("dims", dims, tuple(((dims-1)//2).reshape((len(dims),1))), reference, rps)

    for i, p in enumerate(ps):
        for j, q in enumerate(ps):
            circleindex = np.minimum(np.minimum(np.abs(ps[i]-ps[j]), np.abs(ps[i]-dims-ps[j])), np.abs(ps[i]+dims-ps[j]))
            #print("circleindex", ps[i], ps[j], circleindex, reference)
            targetreference = rps[tuple(((dims-1)//2+circleindex).reshape(len(dims), 1))]

            cds[i, j] = ds[reference, targetreference]
    #print("summation", np.sum(cds, axis=0))
    #assert np.array_equal(np.sum(cds, axis=0), np.ones_like(np.sum(cds, axis=0))*np.max(np.sum(cds, axis=0)))
    return cds




def initmechanisms(mechanisms, d, ps, ds, ep, ws):
    # instance mechanisms with concrete setting
    mechanism_instances = []
    k = 1
    for m in mechanisms:
        if m == 'OCSE':
            # rearrange distance
            # cds = np.copy(ds)
            # for i in range(0, d):
            #     cds[i] = np.roll(ds[math.floor(d/2)], i-math.floor(d/2))
            # set the approximate privacy budget as its median
            # k = math.ceil(d/(np.exp((np.mean(ds)*d*d-d*np.max(ds))/(d*(d-1))*ep)+1))
            cds = circleUp(ds, ps)
            mi = setexponential.KSE(ps, cds, ep, None, m)
            mechanism_instances.append(mi)
        if m == 'ECSE':
            # rearrange distance
            # cds = np.copy(ds)
            # for i in range(0, d):
            #     cds[i] = np.roll(ds[math.floor(d/2)], i-math.floor(d/2))
            # set the approximate privacy budget as its median
            #print("Original Circled Distances", cds)
            cds = circleUp(ds, ps)
            #print("New Circled Distances", cds)
            # k = math.ceil(d/(np.exp((np.mean(ds)*d*d-d*np.max(ds))/(d*(d-1))*ep)+1))
            mi = setexponential.KSE(ps, cds, ep, k, m)
            mechanism_instances.append(mi)
        elif m == 'EKSE':
            # using median privacy budget as approximation
            k = math.ceil(d/(np.exp((np.mean(ds)*d*d-d*np.max(ds))/(d*(d-1))*ep)+1))
            mi = setexponential.KSE(ps, ds, ep, k, m)
            mechanism_instances.append(mi)
        elif m == 'OKSE':
            mi = setexponential.KSE(ps, ds, ep, None, m)
            mechanism_instances.append(mi)
        elif m == 'KSS':
            k = math.ceil(d/(np.exp(np.min(ds[np.nonzero(ds)])*ep)+1))
            mechanism_instances.append(ksubset.KSS(ps, ds, ep, k))
        elif m == 'MRR':
            mechanism_instances.append(mrr.MRR(ps, ds, ep, k))
        elif m == 'BRR':
            mechanism_instances.append(brr.BRR(ps, ds, ep, k))
        elif m == 'GBFMM':
            mechanism_instances.append(gbfmm.GBFMM(ps, ds, ep, k))
        elif m == 'HBFMM':
            mechanism_instances.append(hbfmm.HBFMM(ps, ds, ep, k))
        elif m == 'EM':
            mechanism_instances.append(em.EM(ps, ds, ep, k))
        elif m == 'CEM':
            cds = circleUp(ds, ps)
            mechanism_instances.append(em.EM(ps, cds, ep, k, m))
        elif m == 'BPM':
            mechanism_instances.append(bpm.BPM(ps, ds, ep, ws))
    return mechanism_instances




