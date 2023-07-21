import decimal
from decimal import Decimal as D
import numpy as np
from scipy.special import comb
import numba
from numba import jit
import json
from datetime import datetime, date, time

decimal.getcontext().prec = 8

def Comb(n, k):
    # huge combination numbers using Decimal
    if k < 0 or k > n:
        return D(0.0)
    b = D(1.0)
    for i in range(0, k):
        b = b*D((n-i)/(i+1))
    return b


def PN(K, A, B, C):
    return Comb(n-1, A)*Comb(n-A-1, B)*Comb(n-A-B-1, C)*(K[2,0]**A)*(K[2,1]**B)*(K[2,2]**C)*(K[2,3]**(n-A-B-C-1))


def Pn(K, A, B, C):
    return comb(n-1, A)*comb(n-A-1, B)*comb(n-A-B-1, C)*(K[2,0]**A)*(K[2,1]**B)*(K[2,2]**C)*(K[2,3]**(n-A-B-C-1))


#@jit(nopython=True)
def binarySearchALBHuge4(K, Dab, delta, n, T=10, ER=None):
    EL = 0
    if ER is None:
        ER = Dab

    DK = np.full_like(K, D(0.0), dtype=object)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            DK[i,j] = D(K[i,j])

    K = DK

    for t in range(T):
        Et = (EL+ER)/2.0
        deltat = D(0.0)
        # s=A+B+C
        for s in range(n):
            for A in range(s+1):
                for B in range(s-A+1):
                    C = s-A-B
                    PABC1  = K[0,0]*PN(K, A-1, B, C)+K[0,1]*PN(K, A, B-1, C)+K[0,2]*PN(K, A, B, C-1)+K[0,3]*PN(K, A, B, C)
                    PABC2  = K[1,0]*PN(K, A-1, B, C)+K[1,1]*PN(K, A, B-1, C)+K[1,2]*PN(K, A, B, C-1)+K[1,3]*PN(K, A, B, C)

                    if PABC1 > PABC2*np.exp(Et) or PABC2 > PABC1*np.exp(Et):
                        deltat += PABC2
                    if deltat > delta:
                        break
                if deltat > delta:
                    break
            if deltat > delta:
                break
        if deltat < delta:
            ER = Et
        else:
            EL = Et
        print("Timestamp", t, Et, (EL, ER), deltat, delta)
    return EL, ER


#@jit(nopython=True)
def binarySearchALBFast4(K, Dab, delta, n, T=10, ER=None):
    EL = 0
    if ER is None:
        ER = Dab

    for t in range(T):
        Et = (EL+ER)/2.0
        deltat = 0.0
        # s=A+B+C
        for s in range(n):
            for A in range(s+1):
                for B in range(s-A+1):
                    C = s-A-B
                    PABC1  = K[0,0]*Pn(K, A-1, B, C)+K[0,1]*Pn(K, A, B-1, C)+K[0,2]*Pn(K, A, B, C-1)+K[0,3]*Pn(K, A, B, C)
                    PABC2  = K[1,0]*Pn(K, A-1, B, C)+K[1,1]*Pn(K, A, B-1, C)+K[1,2]*Pn(K, A, B, C-1)+K[1,3]*Pn(K, A, B, C)

                    if PABC1 > PABC2*np.exp(Et) or PABC2 > PABC1*np.exp(Et):
                        deltat += PABC2
                    if deltat > delta:
                        break
                if deltat > delta:
                    break
            if deltat > delta:
                break
        if deltat < delta:
            ER = Et
        else:
            EL = Et
        print("Timestamp", t, Et, (EL, ER), deltat, delta)
    return EL, ER


#@jit(nopython=True)
def binarySearchALBHuge(K, Dab, delta, n, T=30):
    EL = 0
    ER = Dab

    for t in range(T):
        Et = (EL+ER)/2.0
        deltat = D(0.0)
        for s in range(n):
            previousA = 0
            for A in range(s+1):
                Pc = K[2,0]+K[2,1]
                B = s-A

                PAB1  = A*K[0,0]*K[2,1]*(1-Pc)+(s-A)*K[0,1]*K[2,0]*(1-Pc)+(n-s)*(1-K[0,0]-K[0,1])*K[2,0]*K[2,1]
                PAB2  = A*K[1,0]*K[2,1]*(1-Pc)+(s-A)*K[1,1]*K[2,0]*(1-Pc)+(n-s)*(1-K[1,0]-K[1,1])*K[2,0]*K[2,1]

                previousA = A
                if PAB1 > PAB2*np.exp(Et) or PAB2 > PAB1*np.exp(Et):
                    PsA = D(1.0/s)*Comb(n-1, s-1)*(D(K[2,0]+K[2,1])**(s-1))*(D(1-K[2,0]-K[2,1])**(n-s-1))*Comb(s, A)*(D(K[2,0]/(K[2,0]+K[2,1]))**(A-1))*(D(K[2,1]/(K[2,0]+K[2,1]))**(B-1))
                    deltat += D(PAB2)*PsA/D(Pc)
                else:
                    break
            for A in range(s, previousA, -1):
                Pc = K[2,0]+K[2,1]
                B = s-A

                PAB1  = A*K[0,0]*K[2,1]*(1-Pc)+(s-A)*K[0,1]*K[2,0]*(1-Pc)+(n-s)*(1-K[0,0]-K[0,1])*K[2,0]*K[2,1]
                PAB2  = A*K[1,0]*K[2,1]*(1-Pc)+(s-A)*K[1,1]*K[2,0]*(1-Pc)+(n-s)*(1-K[1,0]-K[1,1])*K[2,0]*K[2,1]

                if PAB1 > PAB2*np.exp(Et) or PAB2 > PAB1*np.exp(Et):
                    PsA = D(1.0/s)*Comb(n-1, s-1)*(D(K[2,0]+K[2,1])**(s-1))*(D(1-K[2,0]-K[2,1])**(n-s-1))*Comb(s, A)*(D(K[2,0]/(K[2,0]+K[2,1]))**(A-1))*(D(K[2,1]/(K[2,0]+K[2,1]))**(B-1))
                    deltat += D(PAB2)*PsA/D(Pc)
                else:
                    break
        if deltat < delta:
            ER = Et
        else:
            EL = Et
        print("Timestamp", t, Et, (EL, ER), deltat, delta)
    return EL, ER


#@jit(nopython=True)
def binarySearchALBFast(K, Dab, delta, n, T=30):
    EL = 0
    ER = Dab

    for t in range(T):
        Et = (EL+ER)/2.0
        deltat = 0.0
        for s in range(n):
            for A in range(s+1):
                Pc = K[2,0]+K[2,1]
                Pca = K[2,0]/Pc
                Pcb = K[2,1]/Pc
                B = s-A
                PAB1  = K[0,0]*comb(n-1, s-1)*(Pc**(s-1))*((1-Pc)**(n-s))*comb(s-1, A-1)*(Pca**(A-1))*(Pcb**(B))
                PAB1 += K[0,1]*comb(n-1, s-1)*(Pc**(s-1))*((1-Pc)**(n-s))*comb(s-1, A)*(Pca**A)*(Pcb**(B-1))
                PAB1 += (1-K[0,0]-K[0,1])*comb(n-1, s)*(Pc**s)*((1-Pc)**(n-s-1))*comb(s, A)*(Pca**A)*(Pcb**(B))


                PAB2  = K[1,0]*comb(n-1, s-1)*(Pc**(s-1))*((1-Pc)**(n-s))*comb(s-1, A-1)*(Pca**(A-1))*(Pcb**(B))
                PAB2 += K[1,1]*comb(n-1, s-1)*(Pc**(s-1))*((1-Pc)**(n-s))*comb(s-1, A)*(Pca**A)*(Pcb**(B-1))
                PAB2 += (1-K[1,0]-K[1,1])*comb(n-1, s)*(Pc**s)*((1-Pc)**(n-s-1))*comb(s, A)*(Pca**A)*(Pcb**(B))

                if PAB1 > PAB2*np.exp(Et) or PAB2 > PAB1*np.exp(Et):
                    deltat += PAB2
        if deltat < delta:
            ER = Et
        else:
            EL = Et
        print("Timestamp", t, Et, (EL, ER), deltat, delta)
    return EL, ER


#@jit(nopython=True)
def binarySearchALBFastB(K, Dab, delta, n, T=30):
    EL = 0
    ER = Dab

    for t in range(T):
        Et = (EL+ER)/2.0
        deltat = 0.0
        for s in range(n):
            previousA = 0
            for A in range(s+1):
                Pc = K[2,0]+K[2,1]
                Pca = K[2,0]/Pc
                Pcb = K[2,1]/Pc
                B = s-A
                PAB1  = K[0,0]*comb(n-1, s-1)*(Pc**(s-1))*((1-Pc)**(n-s))*comb(s-1, A-1)*(Pca**(A-1))*(Pcb**(B))
                PAB1 += K[0,1]*comb(n-1, s-1)*(Pc**(s-1))*((1-Pc)**(n-s))*comb(s-1, A)*(Pca**A)*(Pcb**(B-1))
                PAB1 += (1-K[0,0]-K[0,1])*comb(n-1, s)*(Pc**s)*((1-Pc)**(n-s-1))*comb(s, A)*(Pca**A)*(Pcb**(B))


                PAB2  = K[1,0]*comb(n-1, s-1)*(Pc**(s-1))*((1-Pc)**(n-s))*comb(s-1, A-1)*(Pca**(A-1))*(Pcb**(B))
                PAB2 += K[1,1]*comb(n-1, s-1)*(Pc**(s-1))*((1-Pc)**(n-s))*comb(s-1, A)*(Pca**A)*(Pcb**(B-1))
                PAB2 += (1-K[1,0]-K[1,1])*comb(n-1, s)*(Pc**s)*((1-Pc)**(n-s-1))*comb(s, A)*(Pca**A)*(Pcb**(B))

                previousA = A
                if PAB1 > PAB2*np.exp(Et) or PAB2 > PAB1*np.exp(Et):
                    deltat += PAB2
                else:
                    break
            for A in range(s, previousA, -1):
                Pc = K[2,0]+K[2,1]
                Pca = K[2,0]/Pc
                Pcb = K[2,1]/Pc
                B = s-A
                PAB1  = K[0,0]*comb(n-1, s-1)*(Pc**(s-1))*((1-Pc)**(n-s))*comb(s-1, A-1)*(Pca**(A-1))*(Pcb**(B))
                PAB1 += K[0,1]*comb(n-1, s-1)*(Pc**(s-1))*((1-Pc)**(n-s))*comb(s-1, A)*(Pca**A)*(Pcb**(B-1))
                PAB1 += (1-K[0,0]-K[0,1])*comb(n-1, s)*(Pc**s)*((1-Pc)**(n-s-1))*comb(s, A)*(Pca**A)*(Pcb**(B))


                PAB2  = K[1,0]*comb(n-1, s-1)*(Pc**(s-1))*((1-Pc)**(n-s))*comb(s-1, A-1)*(Pca**(A-1))*(Pcb**(B))
                PAB2 += K[1,1]*comb(n-1, s-1)*(Pc**(s-1))*((1-Pc)**(n-s))*comb(s-1, A)*(Pca**A)*(Pcb**(B-1))
                PAB2 += (1-K[1,0]-K[1,1])*comb(n-1, s)*(Pc**s)*((1-Pc)**(n-s-1))*comb(s, A)*(Pca**A)*(Pcb**(B))

                if PAB1 > PAB2*np.exp(Et) or PAB2 > PAB1*np.exp(Et):
                    deltat += PAB2
                else:
                    break
        if deltat < delta:
            ER = Et
        else:
            EL = Et
        print("Timestamp", t, Et, (EL, ER), deltat, delta)
    return EL, ER


#@jit(nopython=True)
def computeK(Dis, name="PEM"):
    # D: local distance metric
    # name: EM, PEM (padded exponential mechanism), or CLONE
    d = 3
    K = None
    if name in ["CLONE"]:
        K = np.zeros((d, d), dtype=float)
        if Dis[0,1] is None:
            p = 1.0
        else:
            p = np.exp(Dis[0,1])/(np.exp(Dis[0,1])+1)
        q = 1.0/(np.exp(Dis[2,0])+np.exp(Dis[2,1]))
        K[0,0] = p
        K[0,1] = 1-p
        K[0,2] = 0
        K[1,0] = 1-p
        K[1,1] = p
        K[1,2] = 0
        K[2,0] = q
        K[2,1] = q
        K[2,2] = 1-2*q
    elif name in ["EM"]:
        K = np.zeros((d, d), dtype=float)
        for i in range(0, d):
            for j in range(0, d):
                K[i,j] = np.exp(-Dis[i,j]/2)
        K = K/np.sum(K, axis=1, keepdims=True)
    else:
        # PEM
        K = np.zeros((d, d+1), dtype=float)
        for i in range(0, d):
            for j in range(0, d):
                K[i,j] = np.exp(-Dis[i,j])
        totalmasses = np.sum(K, axis=1, keepdims=False)
        threshold = 0.0
        for i in range(0, d):
            for j in range(0, d):
                if i != j:
                    summass = (np.exp(Dis[i, j])*np.max([totalmasses[i], totalmasses[j]]) - np.min([totalmasses[i], totalmasses[j]]))/(np.exp(Dis[i, j])-1)
                    if threshold <= summass:
                        threshold = summass
        K[:,d] = threshold - np.sum(K, axis=1, keepdims=False)
        K /= threshold
    print(name, K)
    return K



if __name__ == "__main__":
    epl = 5.0
    #ns = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    ns = [1000, 10000, 100000, 1000000]
    #ns = [1000, 3000, 10000, 30000, 50000]
    #ns = [1000]
    expclones = [i+np.exp(epl)+1 for i in range(0, 10, 1)]+[i+np.exp(epl)+1 for i in range(10, 100, 10)]+[i+np.exp(epl)+1 for i in range(100, 1000, 100)]  # exp(c,a)+exp(c,b)
    #expclones = [i+np.exp(epl)+1 for i in range(50, 500, 100)]  # exp(c,a)+exp(c,b)
    #expclones = [2*np.exp(j) for j in range(1, 7)]

    biasedclone = 2

    results = {}
    results["epl"] = epl
    results["ns"] = ns
    results["expclones"] = expclones
    results["biasedclone"] = biasedclone
    filename = "epl"+str(epl)+"biasedcone"+str(biasedclone)+".json"
    for n in ns:
        results["n"+str(n)] = {}
        results["n"+str(n)]["uppertight"] = []
        results["n"+str(n)]["uppersimpler"] = []
        results["n"+str(n)]["uppersimplest"] = []
        results["n"+str(n)]["uppernumerical"] = []
        results["n"+str(n)]["lowerEM"] = []
        results["n"+str(n)]["lowerPEM"] = []
        for expclone in expclones:
            print("\t\t n,expclone", n, expclone)
            Dis = np.array([
                [0.0, epl, np.log(expclone/2)],
                [epl, 0.0, np.log(expclone/2)],
                [np.log(expclone/2), np.log(expclone/2), 0.0]
            ])

            if biasedclone in [1]:
                # ac=2*bc
                Dis = np.array([
                    [0.0, epl, np.log((np.sqrt(1+4*expclone)-1)/2)*2],
                    [epl, 0.0, np.log((np.sqrt(1+4*expclone)-1)/2)],
                    [np.log((np.sqrt(1+4*expclone)-1)/2)*2, np.log((np.sqrt(1+4*expclone)-1)/2), 0.0]
                ])
            elif biasedclone in [2]:
                # ac=bc+epl
                Dis = np.array([
                    [0.0, epl, np.log(expclone*np.exp(epl)/(np.exp(epl)+1))],
                    [epl, 0.0, np.log(expclone*1/(np.exp(epl)+1))],
                    [np.log(expclone*np.exp(epl)/(np.exp(epl)+1)), np.log(expclone*1/(np.exp(epl)+1)), 0.0]
                ])
            # Dis = np.array([
            #     [0.0, 1.0, 1.0],
            #     [1.0, 0.0, 1.0],
            #     [1.0, 1.0, 0.0]
            # ])
            print("Metric", Dis)

            Dab = Dis[0,1]
            delta = 0.01/n

            twoexp = np.exp(Dis[2,0])+np.exp(Dis[2,1])
            Omega = 2*(n-1)/twoexp-np.sqrt(4*(n-1)*np.log(2/delta)/twoexp)
            UB1 = Dab
            if Omega > 2*np.log(4/delta):
                UB1 = np.log(1+(np.exp(Dab)-1)/(np.exp(Dab)+1)*(2*np.sqrt(np.log(4/delta))+np.sqrt(2/Omega))/(np.sqrt(Omega/2)-np.sqrt(np.log(4/delta))))
            print("Theoretical Upper Bound1", min(UB1, Dab), Omega, 2*np.log(4/delta), twoexp, 2*np.exp(1.0))
            UB1 = min(UB1, Dab)

            twoexp = 2*np.exp(np.max(Dis))
            Omega = 2*(n-1)/twoexp-np.sqrt(4*(n-1)*np.log(2/delta)/twoexp)
            UB2 = Dab
            if n > 16*np.log(4/delta)*np.exp(np.max(Dis)):
                UB2 = np.log(1+(np.exp(Dab)-1)/(np.exp(Dab)+1)*(2*np.sqrt(np.log(4/delta))+np.sqrt(2/Omega))/(np.sqrt(Omega/2)-np.sqrt(np.log(4/delta))))
            print("Theoretical Upper Bound2", min(UB2, Dab), 2*np.log(4/delta), twoexp, 2*np.exp(1.0))
            UB2 = min(UB2, Dab)

            UB3 = Dab
            if n > 16*np.log(4/delta)*np.exp(np.max(Dis)):
                UB3 = np.log(1+(np.exp(Dab)-1)/(np.exp(Dab)+1)*8*(np.sqrt(np.exp(np.max(Dis))*np.log(4/delta)/(n-1))+np.exp(np.max(Dis))/(n-1)))
            print("Theoretical Upper Bound3", min(UB3, Dab), n, 16*np.log(4/delta)*np.exp(np.max(Dis)))
            UB3 = min(UB3, Dab)

            func3 = binarySearchALBFastB
            if n > 1000:
                func3 = binarySearchALBHuge

            T = 20
            """
            UBL, UBR = func3(K, Dab, delta, n, T=15)
            print("Numerical CLONE Upper Bound Fast", UBR)
            """
            K = computeK(Dis, "CLONE")
            import computeamplification as CA
            step = int(0.1*np.power(n, 0.4)+1)
            UBN = CA.numericalanalysis(n, K[2,0]+K[2,1], Dab, delta, T, step, True, coin=K[2,0]/(K[2,0]+K[2,1]), factor=(K[0,0]+K[1,0]))[1]
            numerical_lowerbound = CA.numericalanalysis(n, K[2,0]+K[2,1], Dab, delta, T, step, False, coin=K[2,0]/(K[2,0]+K[2,1]), factor=(K[0,0]+K[1,0]))[1]
            print("Numerical CLONE Upper Bound", UBN, numerical_lowerbound)


            #func4 = binarySearchALBFast4
            #if n > 1000:
            #    func4 = binarySearchALBHuge4

            """
            #K = computeK(Dis, "EM")
            #print("EM Lower Bound", func4(K, Dab, delta, n, T=10))
        
            K = computeK(Dis, "PEM")
            print("PEM Lower Bound Fast", func4(K, Dab, delta, n, T=6, ER=UBR))
            """
            K = computeK(Dis, "EM")
            step = int(0.1*np.power(n, 0.3)+1)
            #numerical_upperbounds = CA.numericalanalysis(n, K[2,0]+K[2,1], Dab, delta, T, step, True, coin=K[2,0]/(K[2,0]+K[2,1]), factor=(K[0,0]+K[1,0]))
            numerical_lowerbounds_em = CA.numericalanalysis(n, K[2,0]+K[2,1], Dab, delta, T, step, False, coin=K[2,0]/(K[2,0]+K[2,1]), factor=(K[0,0]+K[1,0]))
            print("EM Lower Bound", numerical_lowerbounds_em[0], numerical_lowerbounds_em[1])

            K = computeK(Dis, "PEM")
            step = int(0.1*np.power(n, 0.3)+1)
            #numerical_upperbounds = CA.numericalanalysis(n, K[2,0]+K[2,1], Dab, delta, T, step, True, coin=K[2,0]/(K[2,0]+K[2,1]), factor=(K[0,0]+K[1,0]))
            numerical_lowerbounds_cem = CA.numericalanalysis(n, K[2,0]+K[2,1], Dab, delta, T, step, False, coin=K[2,0]/(K[2,0]+K[2,1]), factor=(K[0,0]+K[1,0]))
            print("PEM Lower Bound", numerical_lowerbounds_cem[0], numerical_lowerbounds_cem[1])

            results["n"+str(n)]["uppertight"].append(UB1)
            results["n"+str(n)]["uppersimpler"].append(UB2)
            results["n"+str(n)]["uppersimplest"].append(UB3)
            results["n"+str(n)]["uppernumerical"].append(UBN)
            results["n"+str(n)]["lowerEM"].append(numerical_lowerbounds_em[0])
            results["n"+str(n)]["lowerPEM"].append(numerical_lowerbounds_cem[0])

            #K = computeK(Dis, "PEM")
            #print("PEM Lower Bound Fast", func4(K, Dab, delta, n, T=6, ER=numerical_lowerbound))
    with open(datetime.now().isoformat().replace(':', '_')+'-'+filename, 'w') as outfile:
        json.dump(results, outfile)