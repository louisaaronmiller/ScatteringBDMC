'''
This will follow similar techniques used in NBDMC.py, i.e. function/variable names.

The difference in this version of calculating the scattering length for a 3D
potential square well scattering system is that the scattering ampltiude/
scattering wave equation is written in a different form. Furthermore, it
seeds q',q'' differently, rather than seeding from (q/lambda, q * lambda),
it picks it from the distribution itself, however this distribution is
different depending on two cases that will be evident in the code. Therefore,
this results in different acceptance ratios.

To be aligned with https://files01.core.ac.uk/download/pdf/13617208.pdf, Types
are no longer 0,1,2. They are now A,B,C,D corresponding to each term in
Equation (13) in the paper. Therefore, instead of R12 denoting the acceptance
ratio of Type1 -> Type2, its now RAB to denote movement from A -> B.

New Notation
ZMC = Total number of Monte Carlo states in the simulatioin
ZA = Total number of A-states.
'''
# ============================ PACKAGES ============================

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# ============================ CONSTANTS ============================


# ============================ FUNCTIONS ============================

@njit
def HistogramSum(H):
    '''
    Sums each bin in a histogram, NOTE: it returns the absolute value of each bin summed.
    '''
    return sum([abs(s) for s in H])

@njit
def get_bin(q,deltaq):
    return int(q//deltaq)

@njit
def u_0(potential):
    return (2/3) * potential

@njit
def u(q,potential,eps = 1e-4):
    '''
    For a spherically symmetrical potential that is radius=1
    '''
    if q < eps:
        return u_0(potential)
    else:
        return ((2 * potential)/(q ** 3)) * (np.sin(q) - q * np.cos(q))

@njit
def Iu(qvals,deltaq,potential):
    '''
    This is the integral from 0 to infinity of the absolute value of u(q) dq,
    due to the absolute value, this has no nice analytical expression. Therefore,
    an approximation with sum and infinity going to the max q value multiplied by
    deltaq
    '''
    I_sum = 0
    for q in qvals:
        I_sum += abs(u(q,potential)) * deltaq
    return I_sum

@njit
def If(qvals,deltaq,ZA,H,potential):
    '''
    This is the integral from 0 to infinity of the absolute value of f(q) dq,
    due to the absolute value.
    '''
    return (HistogramSum(H) * Iu(qvals,deltaq,potential) )/ZA

@njit
def Fq(q,ZA,qvals,deltaq,H,potential):
    '''
    q here can be q,qprime,qdprime
    '''
    bin_q = get_bin(q,deltaq)
    return (H[bin_q]/ZA) * Iu(qvals,deltaq,potential)

@njit
def seedchi():
    chi = np.random.uniform(-1,1)
    return chi

@njit
def IndexWeights(weights):
    total = 0
    for i in range(len(weights)):
        total += weights[i]
    if total <= 0:
        return -1
    
    r = np.random.random() * total
    c = 0
    
    for i in range(len(weights)):
        c += weights[i]
        if r <= c:
            return i 
    return weights.shape[0] -1


@njit
def UniBin(bin_index, deltaq):
    '''
    Picks a q uniformly inside bin [bin_index * deltaq, (bin_index + 1)* deltaq]
    '''
    return (bin_index + np.random.random()) * deltaq

@njit
def seedf(H,deltaq):
    '''|f(qprime)|'''
    n = H.shape[0]
    weights = np.empty(n,dtype=np.float64)
    for s in range(n):
        weights[s] = abs(H[s])
    
    s = IndexWeights(weights)

    if s == -1:
        return np.random.random() * (n * deltaq)
    
    return UniBin(s,deltaq)

@njit
def Uweights(qvals,deltaq,potential):
    n = qvals.shape[0]
    w = np.empty(n,dtype=np.float64)

    for i in range(n):
        q = qvals[i]
        if q == 0:
            uq = u_0(potential)
        else:
            uq = u(q,potential)
        w[i] = abs(uq) * deltaq

    return w

@njit
def seedu(qvals,deltaq,potential,u_weights):
    '''|u(qprime)|'''
    i = IndexWeights(u_weights)
    if i == -1:
        return np.random.random() * qvals.shape[0] * deltaq
    return UniBin(i,deltaq)


@njit
def PXY(RXY):
    return min(1,RXY)

@njit
def usqrt(q,q1,chi):
    '''
    sqrt(q^2 +q1^2 - 2qq1chi)
    
    q := q or qprime
    q1 := qprime or qdprime
    '''
    return np.sqrt(q**2 + q1 ** 2 - (2*q*q1*chi))

# ============================ A -> A ============================

@njit
def RAA():
    return 1

# ============================ A -> B, B -> A ============================

@njit
def RAB(q,qprime,PAB,lam,qvals,deltaq,ZA,H,potential,chi):

    numerator1 = 2 * abs(1 + lam) * If(qvals,deltaq,ZA,H,potential)
    denominator1 = np.pi * PAB
    magntiude_term = abs( (u(usqrt(q,qprime,chi),potential)) / (u(q,potential)) )

    answer = (numerator1/denominator1) * magntiude_term
    return answer

@njit
def RBA(q,qprime,PAB,lam,qvals,deltaq,ZA,H,potential,chi):
    return 1/RAB(q,qprime,PAB,lam,qvals,deltaq,ZA,H,potential,chi)

# ============================ A -> C, C -> A ============================

@njit
def RAC(q,qprime,PCA,PAC,lam,qvals,deltaq,potential,chi):
    numerator1 = 2 * abs(lam) * Iu(qvals,deltaq,potential) * PCA
    denominator1 = np.pi * PAC
    magntiude_term = abs( (u(usqrt(q,qprime,chi),potential)) / (u(q,potential)) )
    answer = (numerator1 / denominator1) * magntiude_term
    return answer

@njit
def RCA(q,qprime,PCA,PAC,lam,qvals,deltaq,potential,chi):
    return 1/RAC(q,qprime,PCA,PAC,lam,qvals,deltaq,potential,chi)

# ============================ C -> D, D -> C ============================

@njit
def RCD(qprime,qdprime,PCD,qvals,deltaq,ZA,H,potential,chi):
    '''
    qdprime := q double prime
    '''
    numerator1 = 2 * If(qvals,deltaq,ZA,H,potential)
    denominator1 = np.pi * PCD
    magnitude_term = abs( (u(usqrt(qprime,qdprime,chi),potential)) / (u(qprime,potential)) )
    answer = (numerator1 / denominator1) * magnitude_term
    return answer

@njit
def RDC(qprime,qdprime,PCD,qvals,deltaq,ZA,H,potential,chi):
    return 1/RCD(qprime,qdprime,PCD,qvals,deltaq,ZA,H,potential,chi)

# ============================ HISTOGRAM UPDATER ============================

@njit
def HistogramBuilder(Type,q,qprime,qdprime,chi,potential,ZA_frozen,qvals,deltaq,H_frozen,H_measured):
    
    bin_q = get_bin(q,deltaq)

    if Type == 0:
        sign = np.sign(u(q))
    elif Type == 1:
        sign = np.sign(u(usqrt(q,qprime,chi),potential) * u(qprime,potential))
    elif Type == 2:
        sign = np.sign(u(usqrt(q,qprime,chi),potential) * Fq(qprime,ZA_frozen,qvals,deltaq,H_frozen))
    elif Type == 3:
        sign = np.sign(u(usqrt(q,qprime,chi),potential) * u(usqrt(qprime,qdprime,chi),potential) * Fq(qdprime,ZA_frozen,qvals,deltaq,H_frozen))

    H_measured[bin_q] += sign
    return H_measured


# ============================ SCATTERING APPROX ============================

@njit
def ScatteringApprox(ZA,potential,qvals,deltaq,H):
    multiplier = (2 * Iu(qvals,deltaq,potential))/(np.pi * ZA)
    running_total = 0
    u_0val  = u_0(potential)
    for s in range(len(H)):
        running_total += ( u(qvals[s],potential) * H[s]) 
    term = multiplier * running_total 
    return u_0val + term


# ============================ ALGORITHM ============================
@njit
def BLDMC(PAA,PAB,PAC,PCA,PCD,PBA,PDC,N,M,q0,deltaq,potential,lam): #PBA = PDC = 1
    '''
    DIAGRAM A -> TYPE 0
    DIAGRAM B -> TYPE 1 
    DIAGRAM C -> TYPE 2
    DIAGRAM D -> TYPE 3
    '''
    Type = 0
    Nbins = int(q0/deltaq)
    qvals = np.linspace(0, q0, Nbins)
    u_weights = Uweights(qvals,deltaq,potential)
    H_measured = np.zeros(Nbins)
    H_frozen = np.zeros(Nbins)
    q = seedu(qvals,deltaq,potential,u_weights)

    ZA_frozen = 0
    DiagramAsum = 0
    DiagramBsum = 0
    DiagramCsum = 0
    DiagramDsum = 0

    scattering_length_array = []

    for i in range(M):
        H_measured[:] = 0.0

        for _ in range(N):
            r = np.random.random()

            if Type == 0:
                if r < PAA:
                    r2 = np.random.random()
                    if r2 < RAA():
                        Type = 0 


                elif r < (PAA+PAB):
                    chi= seedchi()
                    qprime = seedf(H_frozen, deltaq)
                    r2 = np.random.random()
                    if r2 < PXY(RAB(q,qprime,PAB,lam,qvals,deltaq,ZA_frozen,H_frozen,potential,chi)):
                        Type = 1

                else: # elif     ->     r < (PAA + PAB + PAC)
                    chi = seedchi()
                    qprime = seedu(qvals,deltaq,potential,u_weights)
                    r2 = np.random.random()
                    if r2 < PXY(RAC(q,qprime,PCA,PAC,lam,qvals,deltaq,potential,chi)):
                        Type = 2
                
            elif Type == 1:

                if r < PBA:
                    r2 = np.random.random()
                    if r2 < PXY(RBA(q,qprime,PAB,lam,qvals,deltaq,ZA_frozen,H_frozen,potential,chi)):
                        Type = 0
            
            elif Type == 2:

                if r < PCA:
                    chi = seedchi()
                    qprime = seedu(qvals,deltaq,potential,u_weights)

                    r2 = np.random.random()
                    if r2 < PXY(RCA(q,qprime,PCA,PAC,lam,qvals,deltaq,potential,chi)):
                        Type = 0
                else: # elif     ->    r < (PCA + PCD)
                    chi= seedchi()
                    qdprime = seedf(H_frozen, deltaq)
                    r2 = np.random.random()
                    if r2 < PXY(RCD(qprime,qdprime,PCD,qvals,deltaq,ZA_frozen,H_frozen,potential,chi)):
                        Type = 3
            
            elif Type == 3:
                if r < PDC:
                    r2 = np.random.random()
                    if r2 < PXY(RDC(qprime,qdprime,PCD,qvals,deltaq,ZA_frozen,H_frozen,potential,chi)):
                        Type = 2
            
            if Type == 0:
                H_measured = HistogramBuilder(0,q,qprime,qdprime,chi,potential,ZA_frozen,qvals,deltaq,H_frozen,H_measured)
                DiagramAsum += 1
            elif Type == 1:
                H_measured = HistogramBuilder(1,q,qprime,qdprime,chi,potential,ZA_frozen,qvals,deltaq,H_frozen,H_measured)
                DiagramBsum += 1
            elif Type == 2:
                H_measured = HistogramBuilder(2,q,qprime,qdprime,chi,potential,ZA_frozen,qvals,deltaq,H_frozen,H_measured)
                DiagramCsum += 1
            elif Type == 3:
                H_measured = HistogramBuilder(3,q,qprime,qdprime,chi,potential,ZA_frozen,qvals,deltaq,H_frozen,H_measured)
                DiagramDsum += 1

        ZA_frozen = DiagramAsum
        H_frozen = H_measured.copy()
        approx = ScatteringApprox(ZA_frozen,potential,qvals,deltaq,H_frozen)
        scattering_length_array.append(approx)
        print(f'Iteration: {i}, Approximation: {round(approx,5)}')

    return scattering_length_array

print(BLDMC(PAA=0.2,PAB=0.4,PAC=0.4,PCA=0.5,PCD=0.5,PBA=1,PDC=1,N = 100000000,M = 30,q0=20,deltaq=0.001,potential=-3,lam=1.5))

