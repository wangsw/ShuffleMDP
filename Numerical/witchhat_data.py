import numpy as np
from scipy import stats
import csv
import warnings
warnings.filterwarnings('ignore')
import scipy.optimize
import random
from math import ceil
from sympy import Symbol, nsolve, log, exp, sqrt
from  tqdm import tqdm
import json
from datetime import datetime, date, time
seed =1
random.seed(seed)
np.random.seed(seed)


# read data
# dataset: https://transparentcalifornia.com/salaries/2011/university-of-california/?&s=base
with open("university-of-california-2011.csv", "r") as f:
    reader = csv.reader(f)
    total_pay = []
    for i, row in enumerate(reader):
        if i > 0:
            total_pay.append(float(row[6]))
true_mean = np.mean(total_pay)
true_max, true_min = np.max(total_pay), np.min(total_pay)
print("true mean: ", true_mean)
print("true max: ", true_max, "true min: ", true_min)
num_data = len(total_pay)
print("num_data: ", num_data)

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

def range_m(B, n, delta, bound="tight"):
    dmin = 0
    dmax = 2 * B
    v = Symbol("v", real=True)
    u = Symbol("u", real=True)
    Gvu = log(1 + 8 * (exp(v) - 1) / (exp(v) + 1) * (sqrt(exp(u) * log(4 / delta) / n) + exp(u) / n))
    tmp = nsolve(16 * log(4 / delta) * exp(v) - n, np.log(n))
    low = B
    high = nsolve(Gvu.subs(u, v) - dmax, dmax)
    realhigh = min(high, tmp)
    high = float(max(realhigh, dmax))
    print("simple tmp", tmp, 16*log(4/delta)*np.exp(dmax), n)
    if bound in ["tight"]:
        v = Symbol("v", real=True)
        Omega = Symbol("Omega", real=True)
        Omegav = n/exp(v)-sqrt(2*n/exp(v)*log(2/delta))
        FvOmega = log(1+(exp(v)-1)/(exp(v)+1)*\
                    (2*sqrt(Omega/2*log(4/delta))+1)/(Omega/2-sqrt(Omega/2*log(4/delta))))
        low = dmax/2
        high2 = msolve(-Omegav/2+sqrt(Omegav/2*log(4/delta)), v, dmax+np.log(n)) #.evalf(chop=True)
        #print("low, high2", low, high2)
        high = msolve(FvOmega.subs(Omega, Omegav)-dmax, v, high2/2) #.evalf(chop=True)
        #print("high, high2", high, high2, (Omegav/2-sqrt(Omegav/2*log(4/delta))).subs(v, high).evalf())
        realhigh = min(high, high2)
        high = max(dmax, realhigh)
        print("tight tmp", (Omegav/2-sqrt(Omegav/2*log(4/delta))).subs(v, dmax).evalf(chop=True), ((2*sqrt(Omega/2*log(4/delta))+1)/(Omega/2-sqrt(Omega/2*log(4/delta)))).subs(Omega, Omegav).subs(v, high).evalf(chop=True), n)
    return float(low), float(high), float(realhigh)

n = 1
shuffle_n = 80000 #259043  # 50000 or num_data #80000
repeats = 10000
B_list = [2]

filename = "numerical_merror_sn"+str(shuffle_n)+"_B"+("-".join([str(v) for v in B_list]))+".json"
# prepare results
results = {}
results["num_data"] = num_data
results["n"] = n
results["shuffle_n"] = shuffle_n
results["B_list"] = list(B_list)

print("n", n, shuffle_n, num_data)
print("B_list: ", B_list)
for B in B_list:
    print("B: ", B)
    # scale data
    begin = np.percentile(total_pay, 5) # minimum
    end = np.percentile(total_pay, 95) # maximum
    maxpay = np.max(total_pay)
    lens = end - begin
    # select num_data from total_pay randomly
    total_pay = list(np.array(total_pay)[np.random.permutation(len(total_pay))[:n]])


    scale_pay = []
    for i in total_pay:
        if i > end:
            scale_pay.append(B)
        elif i < begin:
            scale_pay.append(-B)
        else:
            scale_pay.append(-B + (i - begin) / lens * 2 * B)

    target = np.mean(scale_pay)
    print("scale mean: ", target, begin, end, maxpay)
    n = len(scale_pay)
    delta = 1e-7 #0.01/shuffle_n

    lows = [0.0]*2
    highs = [0.0]*2
    realhighs = [0.0]*2

    lows[0], highs[0], realhighs[0] = range_m(B, shuffle_n, delta, "simple")
    lows[1], highs[1], realhighs[1] = range_m(B, shuffle_n, delta, "tight")
    print("lows, highs, realhighs: ", lows, highs, realhighs)
    low = np.min(lows)
    high = np.max(highs)
    #set m_list according to range_m
    m_list = list(np.arange(low, high, (high-low)/20))
    for v in lows+highs:
        if v not in m_list:
            m_list.append(v)
    m_list = sorted(m_list)
    print("m_list: ", m_list, len(m_list))
    results["B"+str(B)] = {}
    results["B"+str(B)]["mean"] = target
    results["B"+str(B)]["lows"] = lows
    results["B"+str(B)]["highs"] = highs
    results["B"+str(B)]["realhighs"] = realhighs
    results["B"+str(B)]["m_list"] = list(m_list)
    results["B"+str(B)]["bounds"] = ["simple", "tight"]
    results["B"+str(B)]["errors"] = []

    for bound in results["B"+str(B)]["bounds"]:
        Fms = []
        print("bound: ", bound)
        for mi, m in enumerate(m_list):
            F_m = 8*np.sqrt(np.exp(m)*np.log(4/delta)/(shuffle_n-1))+8*np.exp(m)/(shuffle_n - 1)
            if bound in ["tight"]:
                Omega_m = (shuffle_n-1)/np.exp(m)-np.sqrt(2*(shuffle_n-1)*np.log(2/delta)/np.exp(m))
                F_m = (2*np.sqrt(np.log(4/delta))+np.sqrt(2/Omega_m))/(np.sqrt(Omega_m/2)-np.sqrt(np.log(4/delta)))
                if Omega_m <= 2*np.log(2/delta):
                    F_m = 2.0 # Laplace
            Fms.append(F_m)
            output_l = -B - m * F_m / 2
            output_r = B + m * F_m / 2
            N_m = F_m * (1 - np.exp(-m)) + 2 * B * np.exp(-m)
            factor = N_m / (F_m * (1 - np.exp(-m) - m * np.exp(-m)))
            print("m: ", m, F_m)
            #if F_m >= 2.0:
            #    results["B"+str(B)]["errors"].append([0.0, 0.0, 0.0, 0.0])
            #    results["B"+str(B)]["errors"].append([0.0, 0.0, 0.0, 0.0])
            #    continue

            #"""
            class Witchcap(stats.rv_continuous):
                x = 0
                def _pdf(self, z):
                    if np.abs(z - Witchcap.x) <= m * F_m / 2:
                        return np.exp(-2 * np.abs(z - Witchcap.x) / F_m) / N_m
                    else:
                        return np.exp(-m) / N_m


            witchcap = Witchcap(a=output_l, b=output_r)

            laplace_vtve = []
            witchcap_vtve = []
            laplace_vmse =[]
            witchcap_vmse = []
            laplace_tve = []
            witchcap_tve = []
            laplace_mse =[]
            witchcap_mse = []

            for ri in range(repeats):
                lap_noise = np.random.laplace(0,1,size=n)
                laplace_mean = 0.0
                witchcap_mean = 0.0

                laplace_vtve.append(0.0)
                witchcap_vtve.append(0.0)
                laplace_vmse.append(0.0)
                witchcap_vmse.append(0.0)
                for i in range(n):
                    lap_res = scale_pay[i] + lap_noise[i]
                    Witchcap.x = scale_pay[i]
                    witchcap_res = factor * witchcap.rvs()

                    laplace_vtve[-1] += np.abs(lap_res-scale_pay[i])*(end-begin)/(2*B)/n
                    laplace_vmse[-1] += np.square(lap_res-scale_pay[i])*np.power((end-begin)/(2*B), 2.0)/n
                    witchcap_vtve[-1] += np.abs(witchcap_res-scale_pay[i])*(end-begin)/(2*B)/n
                    witchcap_vmse[-1] += np.square(witchcap_res-scale_pay[i])*np.power((end-begin)/(2*B), 2.0)/n

                    laplace_mean += lap_res/n
                    witchcap_mean += witchcap_res/n
                #print("Repeat", ri, laplace_mean, witchcap_mean, target)
                laplace_tve.append(np.abs(laplace_mean-target)*(end-begin)/(2*B))
                witchcap_tve.append(np.abs(witchcap_mean-target)*(end-begin)/(2*B))
                laplace_mse.append(np.square(laplace_mean-target)*np.power((end-begin)/(2*B), 2.0))
                witchcap_mse.append(np.square(witchcap_mean-target)*np.power((end-begin)/(2*B), 2.0))
            results["B"+str(B)]["errors"].append([np.mean(laplace_tve), np.mean(laplace_mse), np.mean(laplace_vtve), np.mean(laplace_vmse), 0.0, 2.0, F_m, 2.0])
            print('laplace (tve, mse, vtve, vmse, tve-theory, mse-theory, F_m, mse-bound): ', results["B"+str(B)]["errors"][-1])
            Witchcap.x = B
            results["B"+str(B)]["errors"].append([np.mean(witchcap_tve), np.mean(witchcap_mse), np.mean(witchcap_vtve), np.mean(witchcap_vmse), 0.0, witchcap.var()*(factor**2), F_m, N_m*(np.exp(-m)*B*(8*B*B+12*m*F_m*B+6*m*m*F_m)+6*(F_m**3)+3*N_m*B*B)/(12*F_m*F_m*((1-np.exp(-m)-m*np.exp(-m))**2))])
            print('witchcap (tve, mse, vtve, vmse, tve-theory, mse-theory F_m, mse-bound): ', results["B"+str(B)]["errors"][-1])
        print('\t\tFms', Fms)
            #"""

with open(datetime.now().isoformat().replace(':', '_')+'-'+filename, 'w') as outfile:
    json.dump(results, outfile)