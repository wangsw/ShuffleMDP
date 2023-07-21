__author__ = 'westbrick'
# simulator for local private discrete distribution estimation

import json
import numpy as np
import scipy as sp
import numpy.random as r

from pyemd import emd

import utils
from distance import *
from datetime import datetime, date, time


class Simulator:
    n = 0  # number of providers
    ds = None  # a list of domain size
    wss = None # weight vectors
    pss = None  # a list of points (topologies)
    dss = None  # a list of distance oracles
    eps = None  # a list of privacy budget epsilons
    histogram = None  # initial histogram
    results = {}  # dict to record simulation settings and results
    mechanisms = None
    repeat = 100  # repeat time for each simulation

    def init(self, n, ds, wss, pss, dss, eps, repeat, mechanisms, histogram=None, shuffle_n=None, debias=True, diameters=None, ampInfo=None):
        self.n = n
        self.shuffle_n = shuffle_n
        self.debias = debias
        self.diameters = diameters
        self.ampInfo = ampInfo
        self.ds = ds
        self.wss = wss
        self.eps = eps
        self.pss = pss
        self.dss = dss
        self.repeat = repeat
        self.mechanisms = mechanisms
        self.histogram = histogram
        self.results['n'] = n
        self.results['shuffle_n'] = shuffle_n
        self.results['debias'] = debias
        self.results['diameters'] = diameters.tolist()
        self.results['ampInfo'] = ampInfo
        self.results['ds'] = ds.tolist()
        self.results['wss'] = [ws.tolist() for ws in wss[0:1]]
        self.results['pss'] = [ps.tolist() for ps in pss[0:1]]
        self.results['dss'] = [ds.tolist() for ds in dss] # record amplified distances
        self.results['eps'] = eps
        self.results['repeat'] = repeat
        #self.results['histogram'] = histogram
        self.results['mechanisms'] = mechanisms
        #print("initial results", self.results)
        for di, d in enumerate(ds):
            self.results['d'+str(di)] = {}
            for ep in eps:
                self.results['d'+str(di)]['ep'+str(ep)] = {}
                #self.results['d'+str(di)]['ep'+str(ep)]['d'] = d
                for m in self.mechanisms:
                    #self.results['d'+str(di)]['ep'+str(ep)]['estimators_'+m] = [None]*self.repeat
                    self.results['d'+str(di)]['ep'+str(ep)]['mean_l2_'+m] = 0.0
                    self.results['d'+str(di)]['ep'+str(ep)]['mean_l1_'+m] = 0.0
                    self.results['d'+str(di)]['ep'+str(ep)]['mean_linf_'+m] = 0.0
                    self.results['d'+str(di)]['ep'+str(ep)]['mean_emd_'+m] = 0.0

                    self.results['d'+str(di)]['ep'+str(ep)]['raw_mean_l2_'+m] = 0.0
                    self.results['d'+str(di)]['ep'+str(ep)]['raw_mean_l1_'+m] = 0.0
                    self.results['d'+str(di)]['ep'+str(ep)]['raw_mean_linf_'+m] = 0.0
                    self.results['d'+str(di)]['ep'+str(ep)]['raw_mean_emd_'+m] = 0.0

                    self.results['d' + str(di)]['ep' + str(ep)]['wmean_l2_' + m] = 0.0
                    self.results['d' + str(di)]['ep' + str(ep)]['wmean_l1_' + m] = 0.0
                    self.results['d' + str(di)]['ep' + str(ep)]['wmean_linf_' + m] = 0.0
                    self.results['d' + str(di)]['ep' + str(ep)]['wmean_emd_' + m] = 0.0

                    self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_l2_' + m] = 0.0
                    self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_l1_' + m] = 0.0
                    self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_linf_' + m] = 0.0
                    self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_emd_' + m] = 0.0

    def simulate(self):
        for di in range(0, len(self.ds)):
            d = self.ds[di]
            ws = self.wss[di]

            for ep in self.eps:
                # initialize mechanisms
                # !!! ep must set to 1.0, without scaling
                mechanism_instances = utils.initmechanisms(self.mechanisms, d, self.pss[di], self.dss[di], ep=1.0, ws=ws)
                #continue
                print('d=', d, 'di=', di, ', epsilon=', ep, ', starts')
                print('metric', self.dss[di][0])
                for rt in range(0, self.repeat):
                    h = self.histogram
                    if h == None:
                        rd = np.array(r.random(d))
                        rd = rd / np.sum(rd)
                        h = utils.histogramer(d, self.n, rd)
                    wh = np.multiply(h, ws)
                    #self.results['d'+str(d)]['ep'+str(ep)]['histograms'][rt] = h.tolist()
                    for m in mechanism_instances:
                        # print('mechanism', m.name)
                        eh = utils.distributor(self.n, h, m, self.debias)
                        weh = np.multiply(eh, ws)
                        #self.results['d'+str(d)]['ep'+str(ep)]['estimators_'+m.name][rt] = eh.tolist()
                        nh = h/self.n
                        neh = eh/self.n
                        nwh = wh / self.n
                        nweh = weh / self.n
                        #print(nh, ws, nwh)
                        self.results['d'+str(di)]['ep'+str(ep)]['raw_mean_l2_'+m.name] += l2norm(nh, neh)
                        self.results['d'+str(di)]['ep'+str(ep)]['raw_mean_l1_'+m.name] += l1norm(nh, neh)
                        self.results['d'+str(di)]['ep'+str(ep)]['raw_mean_linf_'+m.name] += infnorm(nh, neh)
                        self.results['d'+str(di)]['ep'+str(ep)]['raw_mean_emd_'+m.name] += emd(nh, neh, self.dss[0]/self.eps[0])

                        self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_l2_' + m.name] += l2norm(nwh, nweh)
                        self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_l1_' + m.name] += l1norm(nwh, nweh)
                        self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_linf_' + m.name] += infnorm(nwh, nweh)
                        self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_emd_' + m.name] += emd(nwh, nweh, self.dss[0]/self.eps[0])

                        npeh = utils.projector(neh)
                        nwpeh = np.multiply(npeh, ws)

                        #print(nh, npeh, l2norm(nh, npeh))

                        self.results['d'+str(di)]['ep'+str(ep)]['mean_l2_'+m.name] += l2norm(nh, npeh)
                        self.results['d'+str(di)]['ep'+str(ep)]['mean_l1_'+m.name] += l1norm(nh, npeh)
                        self.results['d'+str(di)]['ep'+str(ep)]['mean_linf_'+m.name] += infnorm(nh, npeh)
                        self.results['d'+str(di)]['ep'+str(ep)]['mean_emd_'+m.name] += emd(nh, npeh, self.dss[0]/self.eps[0])

                        self.results['d' + str(di)]['ep' + str(ep)]['wmean_l2_' + m.name] += l2norm(nwh, nwpeh)
                        self.results['d' + str(di)]['ep' + str(ep)]['wmean_l1_' + m.name] += l1norm(nwh, nwpeh)
                        self.results['d' + str(di)]['ep' + str(ep)]['wmean_linf_' + m.name] += infnorm(nwh, nwpeh)
                        self.results['d' + str(di)]['ep' + str(ep)]['wmean_emd_' + m.name] += emd(nwh, nwpeh, self.dss[0]/self.eps[0])

                        """
                        print('  '+str(rt)+'raw_' + m.name, self.results['d' + str(d)]['ep' + str(ep)]['raw_wmean_l2_' + m.name]/(rt+1),
                              self.results['d' + str(d)]['ep' + str(ep)]['raw_wmean_l1_' + m.name]/(rt+1),
                              self.results['d' + str(d)]['ep' + str(ep)]['raw_wmean_linf_' + m.name]/(rt+1))
                        print('  '+str(rt)+'prj_' + m.name, self.results['d' + str(d)]['ep' + str(ep)]['wmean_l2_' + m.name]/(rt+1),
                              self.results['d' + str(d)]['ep' + str(ep)]['wmean_l1_' + m.name]/(rt+1),
                              self.results['d' + str(d)]['ep' + str(ep)]['wmean_linf_' + m.name]/(rt+1))
                        """
                    # print('iteration=', rt, ' ends')


                for m in mechanism_instances:
                    self.results['d'+str(di)]['ep'+str(ep)]['raw_mean_l2_'+m.name] /= self.repeat
                    self.results['d'+str(di)]['ep'+str(ep)]['raw_mean_l1_'+m.name] /= self.repeat
                    self.results['d'+str(di)]['ep'+str(ep)]['raw_mean_linf_'+m.name] /= self.repeat
                    self.results['d'+str(di)]['ep'+str(ep)]['raw_mean_emd_'+m.name] /= self.repeat

                    self.results['d'+str(di)]['ep'+str(ep)]['mean_l2_'+m.name] /= self.repeat
                    self.results['d'+str(di)]['ep'+str(ep)]['mean_l1_'+m.name] /= self.repeat
                    self.results['d'+str(di)]['ep'+str(ep)]['mean_linf_'+m.name] /= self.repeat
                    self.results['d'+str(di)]['ep'+str(ep)]['mean_emd_'+m.name] /= self.repeat

                    self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_l2_' + m.name] /= self.repeat
                    self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_l1_' + m.name] /= self.repeat
                    self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_linf_' + m.name] /= self.repeat
                    self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_emd_' + m.name] /= self.repeat

                    self.results['d' + str(di)]['ep' + str(ep)]['wmean_l2_' + m.name] /= self.repeat
                    self.results['d' + str(di)]['ep' + str(ep)]['wmean_l1_' + m.name] /= self.repeat
                    self.results['d' + str(di)]['ep' + str(ep)]['wmean_linf_' + m.name] /= self.repeat
                    self.results['d' + str(di)]['ep' + str(ep)]['wmean_emd_' + m.name] /= self.repeat

                    # print
                    # print(m.name, self.results['d'+str(d)]['ep'+str(ep)]['wmean_l2_'+m.name], self.results['d'+str(d)]['ep'+str(ep)]['wmean_l1_'+m.name], self.results['d'+str(d)]['ep'+str(ep)]['wmean_linf_'+m.name])
                    print('raw_'+m.name,
                          self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_l2_' + m.name],
                          self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_l1_' + m.name],
                          self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_linf_' + m.name],
                          self.results['d' + str(di)]['ep' + str(ep)]['raw_wmean_emd_' + m.name])
                    print('prj_'+m.name,
                          self.results['d' + str(di)]['ep' + str(ep)]['wmean_l2_' + m.name],
                          self.results['d' + str(di)]['ep' + str(ep)]['wmean_l1_' + m.name],
                          self.results['d' + str(di)]['ep' + str(ep)]['wmean_linf_' + m.name],
                          self.results['d' + str(di)]['ep' + str(ep)]['wmean_emd_' + m.name])
                # print('d=', d, ', epsilon=', ep, ', ends')

    def write(self, filename):
        with open(datetime.now().isoformat().replace(':', '_')+'-'+filename, 'w') as outfile:
            json.dump(self.results, outfile)
        pass

    def read(self, filename):
        with open(filename, 'r') as data_file:
            self.results = json.load(data_file)
        self.n = self.results['n']
        self.ds = self.results['ds']
        self.pss = np.array(self.results['pss'])
        self.dss = np.array(self.results['dss'])
        self.eps = self.results['eps']
        self.repeat = self.results['repeat']













