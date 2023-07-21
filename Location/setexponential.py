__author__ = 'westbrick'

# class for k-subset exponential mechanism

import numpy as np
import scipy as sp
import numpy.random as r
from scipy.special import comb
from numpy.linalg import inv


import utils


class KSE:
    name = 'KSE'
    ep = 0.0    # privacy budget epsilon
    ps = None  # points
    ds = None  # distance oracle
    k = 0  # subset size

    d = 0  # number of points
    rates = None  # matrix of hit rates
    disids = None
    mincrates = None
    crates = None
    u = 0  # size of unique distances
    udis = None  # list of unique distances
    disgroups = None  # matrix of grouped elements for each element
    disgroupsizes = None  # matrix of each group size for each element
    disgroupcounts = None  # matrix of each grouped subset counts  for each element
    disgroupmass = None  # vector of probability mass of each grouped subset counts  for each element
    disgroupaccmass = None  # matrix of accumulate probability masses of each grouped subset counts  for each element
    threshold = 0.0 # unnormalized total probability mass

    def __init__(self, ps, ds, ep, k=None, name=None):
        self.ep = ep
        self.ps = ps
        self.k = k
        self.d = self.ps.shape[0]
        self.ds = np.copy(ds)
        if name != None:
            self.name = name
        if self.k != None:
            self.__setparams()
        else:
            self.selector()
        # print('setexponential', self.ds)

    def __setparams(self):
        self.udis = np.unique(self.ds)
        self.udis = np.sort(self.udis)
        # print(self.udis)
        self.u = len(self.udis)
        self.disids = np.full((self.d, self.d), 0) # disid of d(i,j)
        self.rates = np.full((self.d, self.d), 0.0)
        self.crates = np.full((self.d, self.d, self.d), 0.0)
        self.mincrates = np.full((self.d, self.u), 0.0)
        self.disgroups = np.full((self.d, self.u), None, dtype=list)
        for i in range(0, self.d):
            for j in range(0, self.u):
                self.disgroups[i, j] = []
        self.disgroupsizes = np.full((self.d, self.u), 0.0)
        self.disgroupcounts = np.full((self.d, self.u), 0.0)
        self.disgroupmass = np.full((self.u, ), 0.0)
        self.disgroupaccmass = np.full((self.d, self.u+2), 0.0)

        disgroupaccsizes = np.full((self.d, self.u), 0.0)
        disgroupexp = np.full((self.u, ), 0.0)

        # compute exponential probability mass
        for j in range(0, self.u):
            disgroupexp[j] = np.exp(-self.ep*self.udis[j])

        # group elements by distance for each element
        for i in range(0, self.d):
            for j in range(0, self.d):
                di = 0
                while self.udis[di] != self.ds[i, j]:
                    di += 1
                self.disgroups[i, di].append(j)
        vlen = np.vectorize(len)
        self.disgroupsizes = vlen(self.disgroups)
        # print(self.disgroups, self.disgroupsizes)

        # compute counts by group of each element
        for i in range(0, self.d):
            lefted = self.d
            for j in range(0, self.u):
                disgroupaccsizes[i, j] = lefted
                self.disgroupcounts[i, j] = comb(lefted, self.k)
                lefted -= self.disgroupsizes[i, j]

        for i in range(0, self.d):
            for j in range(0, self.u-1):
                self.disgroupcounts[i, j] -= self.disgroupcounts[i, j+1]
        # print('disgroupcounts', self.disgroupcounts)
        # normalize probabilities
        totalmasses = np.dot(self.disgroupcounts, np.transpose(disgroupexp))
        threshold = 0.0
        for i in range(0, self.d):
            for j in range(0, self.d):
                if i != j:
                    summass = (np.exp(self.ep*self.ds[i, j])*np.max([totalmasses[i], totalmasses[j]]) - np.min([totalmasses[i], totalmasses[j]]))/(np.exp(self.ep*self.ds[i, j])-1)
                    if threshold <= summass:
                        threshold = summass
        self.threshold = threshold
        self.disgroupmass = disgroupexp/self.threshold

        # compute accumulate probability masses
        for i in range(0, self.d):
            last = 0.0
            for j in range(0, self.u):
                self.disgroupaccmass[i, j] = last
                last = last+self.disgroupmass[j]*self.disgroupcounts[i, j]
            self.disgroupaccmass[i, self.u] = last
            self.disgroupaccmass[i, self.u+1] = 1.0

        # compute hitrates
        hitcounts = np.full((self.d, self.d, self.u), 0.0)
        for i in range(0, self.d):
            for l in range(0, self.d):
                for j in range(0, self.u):
                    if self.udis[j] <= self.ds[i, l]:
                        hitcounts[i, l, j] = sp.special.binom(disgroupaccsizes[i, j]-1, self.k-1)
                for j in range(0, self.u-1):
                    hitcounts[i, l, j] = hitcounts[i, l, j] - hitcounts[i, l, j+1]

        for i in range(0, self.d):
            self.rates[i] = np.dot(hitcounts[i], self.disgroupmass)

        # print(self.ps)
        # print(self.disgroups)
        # print(self.disgroupcounts)
        #print("k-rates", self.name, self.k, self.rates, inv(self.rates))

    def selector(self):
        # select error-bound optimal k
        ks = list(range(1, self.d))
        bounds = np.full((len(ks),), 0.0)
        self.udis = np.unique(self.ds)
        self.udis = np.sort(self.udis)
        # print(self.udis)
        self.u = len(self.udis)
        self.disids = np.full((self.d, self.d), 0) # disid of d(i,j)
        self.disgroups = np.full((self.d, self.u), None, dtype=list)
        self.disgroupsizes = np.full((self.d, self.u), 0.0)
        for i in range(0, self.d):
            for j in range(0, self.u):
                self.disgroups[i, j] = []

        disgroupexp = np.full((self.u, ), 0.0)

        # compute exponential probability mass
        for j in range(0, self.u):
            disgroupexp[j] = np.exp(-self.ep*self.udis[j])

        # group elements by distance for each element
        for i in range(0, self.d):
            for j in range(0, self.d):
                di = 0
                while self.udis[di] != self.ds[i, j]:
                    di += 1
                self.disgroups[i, di].append(j)
                self.disids[i, j] = di
        vlen = np.vectorize(len)
        self.disgroupsizes = vlen(self.disgroups)
        #print(self.disgroups, self.disgroupsizes)

        for k in ks:
            # compute counts by group of each element
            self.k = k
            self.rates = np.full((self.d, self.d), 0.0)
            self.crates = np.full((self.d, self.d, self.d), 0.0)
            self.mincrates = np.full((self.d, self.u), 0.0)
            self.disgroupcounts = np.full((self.d, self.u), 0.0)
            self.disgroupmass = np.full((self.u, ), 0.0)
            self.disgroupaccmass = np.full((self.d, self.u+2), 0.0)

            disgroupaccsizes = np.full((self.d, self.u), 0.0)


            for i in range(0, self.d):
                lefted = self.d
                for j in range(0, self.u):
                    disgroupaccsizes[i, j] = lefted
                    self.disgroupcounts[i, j] = comb(lefted, self.k)
                    #print('comb', lefted, self.k, comb(lefted, self.k))
                    lefted -= self.disgroupsizes[i, j]
                    #print('disgroupsizes', self.disgroupsizes[i, j])

            for i in range(0, self.d):
                for j in range(0, self.u-1):
                    self.disgroupcounts[i, j] -= self.disgroupcounts[i, j+1]
            #print('disgroupcounts', self.disgroupcounts)
            # normalize probabilities
            totalmasses = np.dot(self.disgroupcounts, np.transpose(disgroupexp))
            threshold = 0.0
            for i in range(0, self.d):
                # print(i, totalmasses[i])
                for j in range(0, self.d):
                    if i != j:
                        summass = (np.exp(self.ep*self.ds[i, j])*np.max([totalmasses[i], totalmasses[j]]) - np.min([totalmasses[i], totalmasses[j]]))/(np.exp(self.ep*self.ds[i, j])-1)
                        if threshold <= summass:
                            threshold = summass
            self.threshold = threshold
            self.disgroupmass = disgroupexp/self.threshold
            #print('threshold', self.threshold)
            # compute accumulate probability masses
            for i in range(0, self.d):
                last = 0.0
                for j in range(0, self.u):
                    self.disgroupaccmass[i, j] = last
                    last = last+self.disgroupmass[j]*self.disgroupcounts[i, j]
                self.disgroupaccmass[i, self.u] = last
                self.disgroupaccmass[i, self.u+1] = 1.0

            # compute hitrates
            """
            hitcounts = np.full((self.d, self.d, self.u), 0.0)
            for i in range(0, self.d):
                for l in range(0, self.d):
                    for j in range(0, self.u):
                        if self.udis[j] <= self.ds[i, l]:
                            hitcounts[i, l, j] = sp.special.binom(disgroupaccsizes[i, j]-1, self.k-1)
                    for j in range(0, self.u-1):
                        hitcounts[i, l, j] = hitcounts[i, l, j] - hitcounts[i, l, j+1]

            for i in range(0, self.d):
                self.rates[i] = np.dot(hitcounts[i], self.disgroupmass)
            """
            for i in range(0, self.d):
                res = self.d-1
                former = sp.special.binom(res, self.k-1)
                self.rates[i, i] = former/self.threshold
                #print("threshold", res, self.k, former, self.threshold)
                partmass = 0.0
                for j in range(0, self.u-1):
                    res = res-self.disgroupsizes[i, j]
                    partmass += (former-sp.special.binom(res, self.k-1))*self.disgroupmass[j]
                    former = sp.special.binom(res, self.k-1)
                    for l in self.disgroups[i, j+1]:
                        self.rates[i, l] = partmass + former*self.disgroupmass[j+1]
            #print("k-rates-selector", self.ep, self.name,  self.k, self.rates)


            # compute trimmed ch[c,a,b]
            for i in range(0, self.d):
                res = self.d-2
                former = sp.special.binom(res, self.k-2)
                #print('ch', sp.special.binom(res, self.k-2))
                self.mincrates[i, 0] = former/self.threshold
                partmass = 0.0
                for j in range(0, self.u-1):
                    res = res-self.disgroupsizes[i, j]
                    if res >= 0:
                        partmass += (former-sp.special.binom(res, self.k-2))*self.disgroupmass[j]
                        former = sp.special.binom(res, self.k-2)
                    else:
                        former = 0
                    self.mincrates[i, j+1] = partmass + former*self.disgroupmass[j+1]
                    #print('chs', res, self.k-2, (sp.special.binom(res, self.k-2)))
            #print("mincrates", self.name, self.k, self.u, self.mincrates)


            # compute <ih_a, ih_b>
            ih = inv(self.rates)
            hab = np.full((self.d, self.d), 0.0)
            for a in range(0, self.d):
                for b in range(0, self.d):
                    hab[a, b] = np.dot(ih[a], ih[b])

            # compute MSE
            mse = 0.0
            for a in range(0, self.d):
                for b in range(0, self.d):
                    cmass = 0.0
                    for c in range(0, self.d):
                        if a != b:
                            cmass += 1/self.d*(self.mincrates[c, min(self.disids[c, a], self.disids[c, b])]-self.rates[c, a]*self.rates[c, b])
                        else:
                            cmass += 1/self.d*(self.rates[c, a]-self.rates[c, a]*self.rates[c, b])
                    mse += cmass*hab[a, b]
            bounds[ks.index(k)] = mse

        print('k-bounds', self.ep, self.name, self.k, bounds)
        minki = int(np.where(bounds == np.min(bounds))[0])

        self.k = ks[minki]
        #self.k = 2
        print('k-opt', self.name, self.k)
        self.__setparams()



    def randomizer(self, secret, k=None):
        pub = []
        if k == None:
            k = self.k
        p = r.random(1)

        # print(self.disgroupaccmass[secret])
        mini = utils.binarysearch(self.disgroupaccmass[secret], p)
        # print(secret, p, mini)

        if mini >= self.u:
            if self.name in ["OCSE", "ECSE"]:
                assert False
            return pub

        # union groups
        eqset = self.disgroups[secret, mini]
        gtset = []
        for j in range(mini+1, self.u):
            gtset.extend(self.disgroups[secret, j])
        # print(eqset, gtset)

        # conditions
        conditions = np.full((np.min([len(eqset), k]),), 0.0)
        for i in range(0, len(conditions)):
            conditions[i] = sp.special.binom(len(eqset), i+1)*sp.special.binom(len(gtset), k-(i+1))
        conditions = conditions/np.sum(conditions)

        accconditions = np.full((np.min([len(eqset), k])+1,), 0.0)
        last = 0.0
        for i in range(0, len(conditions)):
            accconditions[i] = last
            last += conditions[i]
        accconditions[len(conditions)] = last
        # sample
        q = r.random(1)
        mins = utils.binarysearch(accconditions, q)+1

        pub.extend(utils.reservoirsample(eqset, mins))
        pub.extend(utils.reservoirsample(gtset, k-mins))

        return pub

    def decoder(self, hits, n, debias=True, truncate=False, threshold=0.05):
        fs = hits
        # debias hits but without projecting to simplex
        rates = np.copy(self.rates)
        if truncate:
            rates[np.where(rates<threshold)] = 0.0
        if debias:
            fs = np.dot(hits, inv(rates))
        #print("inv(rates, self.rates)", inv(rates), inv(self.rates))
        return fs

    def bound(self, n, tfs=None):
        # estimate theoretical l2-norm error bound
        varobsv = self.rates*(1.0-self.rates)
        invfm = self.rates.I
        sqrfm = invfm*invfm
        cps = np.full((self.d, self.d, self.d), 0.0)
        for i in range(0, self.d):
            cps[i] = np.dot(self.rates[i], self.rates[i].T)
        cps = cps*np.power(1.7*(self.k-1)/self.d, 0.4)    # empirical formula

        bound = np.sum(np.dot(tfs, np.dot(varobsv, sqrfm)))
        for m in range(0, self.d):
            for l in range(0, self.d):
                for i in range(0, self.d):
                    for j in range(0, self.d):
                        if j != i:
                            bound += tfs[l](cps[l, i, j]-self.rates[l, i]*self.rates[l, j])*invfm[i, m]*invfm[j, m]
        return bound/(n*n)





































