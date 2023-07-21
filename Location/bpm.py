__author__ = 'westbrick'
# class for bi-parties mechanism

import numpy as np
import numpy.random as r


class BPM:
    name = 'BPM'
    ep = 0.0  # privacy budget epsilon
    ps = None  # points
    ds = None  # distance oracle

    d = 0  # number of points
    k = 0 # partition bound
    eph = 0.0 # privacy budget for heavyweight items

    htrate = 0.0
    hfrate = 0.0
    ltrate = 0.0
    lfrate = 0.0

    def __init__(self, ps, ds, ep, ws, k=None):
        self.ep = ep
        self.ps = ps
        self.ds = np.copy(ds)
        self.d = self.ps.shape[0]
        self.ws = ws
        self.k = k
        self.eph = 0.5*self.ep
        #print('brr', self.ds)
        self.__setparams()

    def __setparams(self):
        # find optimal k and eph
        ks = []
        if self.k == None:
            ks = range(1, self.d+1)
        else:
            ks = [self.k]

        # multiple round iterative Newton's method

        minf = None
        for k in ks:
            eph = 0.75*self.ep
            while True:
                ephnow = eph
                dfl =  [-self.ws[i]*self.ws[i]*(np.exp(eph)*(1.0/self.d*(k-1)*(np.exp(eph)-1)+1.0*(np.exp(eph)+2*k-1)))/(np.power(np.exp(eph)-1,3)) for i in range(0,k)]
                dfl += [self.ws[i]*self.ws[i]*(1.0*np.exp(self.ep+eph)*(np.exp(self.ep)+np.exp(eph)))/(np.power(np.exp(self.ep)-np.exp(eph),3)) for i in range(k,self.d)]
                df = sum(dfl)
                ddfl = [self.ws[i]*self.ws[i]*np.exp(eph)*(1.0/self.d*(np.exp(2*eph)-1)*(k-1)+1.0*(np.exp(2*eph)+4*np.exp(eph)*k+2*k-1))/(np.power(np.exp(eph)-1,4)) for i in range(0, k)]
                ddfl+= [self.ws[i]*self.ws[i]*1.0*np.exp(self.ep)*np.exp(eph)*(np.exp(2*self.ep)+np.exp(2*eph)+4*np.exp(self.ep+eph))/(np.power(np.exp(self.ep)-np.exp(eph),4)) for i in range(k,self.d)]
                ddf = sum(ddfl)
                eph = ephnow-(df)/(ddf)
                if np.abs(ephnow-eph)<0.001*self.ep or eph <= 0.5*self.ep or eph >= self.ep:
                    break
            #print(eph)
            eph = min(max(eph, 0.5*self.ep), self.ep)
            fl = [self.ws[i]*self.ws[i]*(1.0/self.d*np.exp(eph)*k+(1.0-1.0/self.d)*(np.exp(eph)+k-1))/(np.power(np.exp(eph)-1.0,2)) for i in range(0, k)]
            fl +=[self.ws[i]*self.ws[i]*(1.0*np.exp(self.ep-eph))/(np.power(np.exp(self.ep-eph)-1,2)) for i in range(k, self.d)]
            #print(fl)
            f = sum(fl)
            #print('Iteration', k, eph, f)
            if minf==None or f < minf:
                self.k = k
                self.eph = eph
                minf = f
        print('BPM', self.k, self.eph)
        self.htrate = (np.exp(self.eph)) / (np.exp(self.eph) + self.k)
        self.hfrate = (1.0) / (np.exp(self.eph) + self.k)
        self.ltrate = (np.exp(self.ep - self.eph)) / (np.exp(self.ep - self.eph) + 1)
        self.lfrate = (1.0) / (np.exp(self.ep - self.eph) + 1)


    def randomizer(self, secret):
        pub = []
        if secret < self.k:
            selection = secret
            p = r.random(1)[0]
            if p > self.htrate - self.hfrate:
                selection = r.choice(self.k+1, 1)[0]
            if selection != self.k:
                pub = [selection]
            else:
                pub = []
            for i in range(self.k, self.d):
                p = r.random(1)[0]
                if p < self.lfrate:
                    pub.append(i)
        else:
            selection = self.k
            p = r.random(1)[0]
            if p > self.htrate - self.hfrate:
                selection = r.choice(self.k+1, 1)[0]
            if selection != self.k:
                pub = [selection]
            else:
                pub = []
            for i in range(self.k, self.d):
                p = r.random(1)[0]
                if i == secret:
                    if p < self.ltrate:
                        pub.append(i)
                else:
                    if p < self.lfrate:
                        pub.append(i)
        #print(secret, pub)
        return pub

    def decoder(self, hits, n, debias=True):
        fs = hits
        if debias:
            # debias hits but without projecting to simplex
            fs = np.array([(hits[i]-n*self.hfrate)/(self.htrate-self.hfrate) for i in range(0, self.k)]+[(hits[i]-n*self.lfrate)/(self.ltrate-self.lfrate) for i in range(self.k, self.d)])
        # print(fs)
        return fs

    def bound(self, n, tfs=None):
        # compute theoretical l2-norm error bound
        # return (self.htrate*(1.0-self.htrate)+self.k*self.hfrate*(1-self.hfrate))/(n*(self.htrate-self.hfrate)*(self.htrate-self.hfrate))
        return 0.0





