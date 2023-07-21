__author__ = 'westbrick'
# class for greedy bit flipping matrix mechanism

import numpy as np
import numpy.random as r


class GBFMM:
    name = 'GBFMM'
    ep = 0.0    # privacy budget epsilon
    ps = None  # points
    ds = None  # distance oracle

    d = 0 # number of points
    trates = [] # hit rate when true
    frates = [] # hit rate when false

    def __init__(self, ps, ds, ep, k=None):
        self.ep = ep
        self.ps = ps
        self.ds = np.copy(ds)
        self.d = self.ps.shape[0]
        self.trates = [0.0]*self.d
        self.frates = [0.0]*self.d
        self.__setparams()
        #print('gbfmm', self.ds)

    def __setparams(self):
        for i in range(0, self.d):
            mindis = np.max(self.ds[i])
            for j in range(0, self.d):
                if j != i and mindis > self.ds[i, j]:
                    mindis = self.ds[i, j]
            self.trates[i] = np.exp(self.ep*mindis/2)/(np.exp(self.ep*mindis/2)+1)
            self.frates[i] = 1.0/(np.exp(self.ep*mindis/2)+1)
        # print('GBFMM', self.trates, self.frates)

    def randomizer(self, secret):
        pub = []
        for i in range(0, self.d):
            p = r.random(1)
            if i == secret:
                if p < self.trates[i]:
                    pub.append(i)
            else:
                if p < self.frates[i]:
                    pub.append(i)
        return pub

    def decoder(self, hits, n):
        # debiaing hits but without projecting to simplex
        fs = np.array([(hits[i]-n*self.frates[i])/(self.trates[i]-self.frates[i]) for i in range(0, self.d)])
        return fs

    def bound(self, n, tfs=None):
        # compute theoretical l2-norm error bound
        bound = 0.0
        if tfs == None:
            tfs = np.full((self.d,), 1.0/self.d)
        for i in range(0, self.d):
            bound += (tfs[i]*self.trates[i]*(1.0-self.trates[i])+(n-tfs[i])*self.frates[i]*(1-self.frates[i]))/(n*(self.trates[i]-self.frates[i])*(self.trates[i]-self.frates[i]))
        return bound/(n*n)