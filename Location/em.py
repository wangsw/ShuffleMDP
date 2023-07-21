__author__ = 'westbrick'

__author__ = 'westbrick'
# class for multivariate randomized response

import numpy as np
import numpy.random as r
from numpy.linalg import inv

class EM:
    name = 'EM'
    ep = 0.0    # privacy budget epsilon
    ps = None  # points
    ds = None  # distance oracle

    d = 0  # number of points
    rates = None  # transition probability matrix

    def __init__(self, ps, ds, ep, k=None, name=None):
        self.ep = ep
        self.ps = ps
        self.ds = np.copy(ds)
        self.d = self.ps.shape[0]
        #print('brr', self.ds)
        if name != None:
            self.name = name
        self.__setparams()

    def __setparams(self):
        self.rates = np.full((self.d, self.d), 0.0)
        for i in range(0, self.d):
            for j in range(0, self.d):
                factor = 2
                if self.name in ["CEM"]:
                    factor = 1
                self.rates[i, j] = np.exp(-self.ep*self.ds[i, j]/factor)
        for i in range(0, self.d):
            total = np.sum(self.rates[i])
            self.rates[i] = self.rates[i]/total


    def randomizer(self, secret):
        p = r.random(1)
        pub = 0
        while p > self.rates[secret, pub]:
            p -= self.rates[secret, pub]
            pub += 1
        return [pub]

    def decoder(self, hits, n, debias=True):
        fs = hits
        if debias:
            # debias hits but without projecting to simplex
            fs = np.dot(hits, inv(self.rates))
        return fs

    def bound(self, n, tfs=None):
        # compute theoretical l2-norm error bound
        return 0.0









