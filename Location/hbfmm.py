__author__ = 'westbrick'

# class for heuristic bit flipping matrix mechanism

import numpy as np
import numpy.random as r


class HBFMM:
    name = 'HBFMM'
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
        #print('hbfmm', self.ds)

    def __setparams(self):
        cds = np.copy(self.ds)
        for i in range(0, self.d):
            cds[i, i] = np.max(cds)

        handled = []
        while len(handled) != self.d:
            mindis = np.max(cds[i])
            mini = -1
            minj = -1
            for i in range(0, self.d):
                if i not in handled:
                    for j in range(0, self.d):
                        if cds[i, j] <= mindis:
                            mindis = cds[i, j]
                            mini = i
                            minj = j

            i = mini
            if i not in handled:
                self.trates[i] = np.exp(self.ep*mindis/2)/(np.exp(self.ep*mindis/2)+1)
                self.frates[i] = 1.0/(np.exp(self.ep*mindis/2)+1)
                handled.append(i)
                for j in range(0, self.d):
                    if j not in handled:
                        cds[j, mini] = 2*cds[j, mini]-mindis
            i = minj
            if i not in handled:
                self.trates[i] = np.exp(self.ep*mindis/2)/(np.exp(self.ep*mindis/2)+1)
                self.frates[i] = 1.0/(np.exp(self.ep*mindis/2)+1)
                handled.append(i)
                for j in range(0, self.d):
                    if j not in handled:
                        cds[j, minj] = 2*cds[j, minj]-mindis
        # print('HBFMM', self.trates, self.frates)

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

    def decoder(self, hits, n, debias=True):
        fs = hits
        if debias:
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
