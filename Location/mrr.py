__author__ = 'westbrick'
# class for multivariate randomized response

import numpy as np
import numpy.random as r


class MRR:
    name = 'MRR'
    ep = 0.0    # privacy budget epsilon
    ps = None  # points
    ds = None  # distance oracle

    dis = 0.0  # distance between two points
    d = 0 # number of points
    trate = 0.0 # hit rate when true
    frate = 0.0 # hit rate when false

    def __init__(self, ps, ds, ep, k=None):
        self.ep = ep
        self.ps = ps
        self.ds = np.copy(ds)
        self.d = self.ps.shape[0]
        self.dis = np.min(self.ds[np.nonzero(self.ds)])
        #print('mrr', self.ds)
        self.__setparams()

    def __setparams(self):
        self.trate = np.exp(self.ep*self.dis)/(np.exp(self.ep*self.dis)+self.d-1)
        self.frate = 1.0/(np.exp(self.ep*self.dis)+self.d-1)

    def randomizer(self, secret):
        pub = secret
        p = r.random(1)
        if p > self.trate-self.frate:
            pub = r.choice(self.d, 1)

        return [pub]

    def decoder(self, hits, n, debias=True):
        fs = hits
        if debias:
            # debias hits but without projecting to simplex
            fs = np.array([(hits[i]-n*self.frate)/(self.trate-self.frate) for i in range(0, self.d)])
        return fs

    def bound(self, n, tfs=None):
        # compute theoretical l2-norm error bound
        return (self.trate*(1.0-self.trate)+(self.d-1)*self.frate*(1-self.frate))/(n*(self.trate-self.frate)*(self.trate-self.frate))









