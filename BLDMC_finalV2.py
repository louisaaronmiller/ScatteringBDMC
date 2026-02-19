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

IMPORTANT
Iu is the function that generates I_u to be used
this is because Iu was constantly called upon and made the runtime very long

OPTIMIZATION:
1) Cache If_frozen once per outer iteration (instead of recomputing HistogramSum inside acceptance ratios).
2) Precompute CDF for |H_frozen| once per outer iteration (seedf becomes O(log Nbins)).
3) Precompute CDF for |u(q)| weights once per run (seedu becomes O(log Nbins)).
'''

# ============================ PACKAGES ============================

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# ============================ FUNCTIONS ============================

@njit
def HistogramSum(H):
    '''
    Sums each bin in a histogram, NOTE: it returns the absolute value of each bin summed.
    '''
    s = 0.0
    for i in range(H.shape[0]):
        s += abs(H[i])
    return s

@njit
def get_bin(q, deltaq):
    return int(q // deltaq)

@njit
def u_0(potential):
    return (2/3) * potential

@njit
def u(q, potential, eps=1e-4):
    '''
    For a spherically symmetrical potential that is radius=1
    '''
    if q < eps:
        return u_0(potential)
    else:
        return ((2 * potential) / (q ** 3)) * (np.sin(q) - q * np.cos(q))

@njit
def Iu(qvals, deltaq, potential):
    '''
    This is the integral from 0 to infinity of the absolute value of u(q) dq,
    due to the absolute value, this has no nice analytical expression. Therefore,
    an approximation with sum and infinity going to the max q value multiplied by
    deltaq
    '''
    I_sum = 0.0
    for i in range(qvals.shape[0]):
        q = qvals[i]
        I_sum += abs(u(q, potential)) * deltaq
    return I_sum

@njit
def If(qvals, deltaq, ZA, H, potential, I_u):
    '''
    This is the integral from 0 to infinity of the absolute value of f(q) dq,
    due to the absolute value.
    '''
    return (HistogramSum(H) * I_u) / ZA

@njit
def Fq(q, ZA, qvals, deltaq, H, potential, I_u):
    '''
    q here can be q,qprime,qdprime
    '''
    bin_q = get_bin(q, deltaq)
    return (H[bin_q] / ZA) * I_u

@njit
def seedchi():
    chi = np.random.uniform(-1, 1)
    return chi

@njit
def IndexWeights(weights):
    total = 0.0
    for i in range(len(weights)):
        total += weights[i]
    if total <= 0:
        return -1

    r = np.random.random() * total
    c = 0.0

    for i in range(len(weights)):
        c += weights[i]
        if r <= c:
            return i
    return weights.shape[0] - 1

@njit
def UniBin(bin_index, deltaq):
    '''
    Picks a q uniformly inside bin [bin_index * deltaq, (bin_index + 1)* deltaq]
    '''
    return (bin_index) * deltaq # + np.random.random() (added to bin_index, however I think this is wrong)

@njit
def seedf(H, deltaq):
    '''|f(qprime)|  (original slow version kept for reference)'''
    n = H.shape[0]
    weights = np.empty(n, dtype=np.float64)
    for s in range(n):
        weights[s] = abs(H[s])

    s = IndexWeights(weights)

    if s == -1:
        return np.random.random() * (n * deltaq)

    return UniBin(s, deltaq)

@njit
def Uweights(qvals, deltaq, potential):
    n = qvals.shape[0]
    w = np.empty(n, dtype=np.float64)

    for i in range(n):
        q = qvals[i]
        if q == 0:
            uq = u_0(potential)
        else:
            uq = u(q, potential)
        w[i] = abs(uq) * deltaq

    return w

@njit
def seedu(qvals, deltaq, potential, u_weights):
    '''|u(qprime)|  (original slow version kept for reference)'''
    i = IndexWeights(u_weights)
    if i == -1:
        return np.random.random() * qvals.shape[0] * deltaq
    return UniBin(i, deltaq)

@njit
def PXY(RXY):
    return min(1.0, RXY)

@njit
def usqrt(q, q1, chi):
    '''
    sqrt(q^2 +q1^2 - 2qq1chi)
    q := q or qprime
    q1 := qprime or qdprime
    '''
    return np.sqrt(q**2 + q1**2 - (2.0*q*q1*chi))

# ============================ FAST CDF SAMPLING ============================

@njit
def BuildCDF(weights, cdf_out):
    """
    cdf_out[i] = sum_{j<=i} weights[j]
    returns total weight
    """
    total = 0.0
    for i in range(weights.shape[0]):
        total += weights[i]
        cdf_out[i] = total
    return total

@njit
def SampleCDF(cdf, total):
    """
    Returns index sampled proportional to weights that generated `cdf`.
    Binary search -> O(log Nbins)
    """
    if total <= 0.0:
        return -1

    r = np.random.random() * total
    lo = 0
    hi = cdf.shape[0] - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if r <= cdf[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

@njit
def SeedFromCDF(cdf, total, deltaq):
    i = SampleCDF(cdf, total)
    if i == -1:
        return np.random.random() * (cdf.shape[0] * deltaq)
    return UniBin(i, deltaq)

# ============================ A -> A ============================

@njit
def RAA():
    return 1.0

# ============================ A -> B, B -> A (FAST If) ============================

@njit
def RAB_fast(q, qprime, PAB, lam, If_frozen, potential, chi):
    numerator1 = 2.0 * abs(1.0 + lam) * If_frozen
    denominator1 = np.pi * PAB
    magnitude_term = abs(u(usqrt(q, qprime, chi), potential) / u(q, potential))
    return (numerator1 / denominator1) * magnitude_term

# ============================ A -> C, C -> A ============================

@njit
def RAC(q, qprime, PCA, PAC, lam, qvals, deltaq, potential, chi, I_u):
    numerator1 = 2.0 * abs(lam) * I_u * PCA
    denominator1 = np.pi * PAC
    magnitude_term = abs(u(usqrt(q, qprime, chi), potential) / u(q, potential))
    return (numerator1 / denominator1) * magnitude_term

@njit
def RCA(q, qprime, PCA, PAC, lam, qvals, deltaq, potential, chi, I_u):
    return 1.0 / RAC(q, qprime, PCA, PAC, lam, qvals, deltaq, potential, chi, I_u)

# ============================ C -> D, D -> C (FAST If) ============================

@njit
def RCD_fast(qprime, qdprime, PCD, If_frozen, potential, chi):
    numerator1 = 2.0 * If_frozen
    denominator1 = np.pi * PCD
    magnitude_term = abs(u(usqrt(qprime, qdprime, chi), potential) / u(qprime, potential))
    return (numerator1 / denominator1) * magnitude_term

# ============================ HISTOGRAM UPDATER ============================

@njit
def HistogramBuilder(Type, q, qprime, qdprime, chi, potential,
                     ZA_frozen, qvals, deltaq, H_frozen, H_measured, I_u, chip):

    bin_q = get_bin(q, deltaq)

    if Type == 0:
        sign = -np.sign(u(q, potential))
    elif Type == 1:
        sign = -np.sign(u(usqrt(q, qprime, chi), potential) *
                        Fq(qprime, ZA_frozen, qvals, deltaq, H_frozen, potential, I_u))
    elif Type == 2:
        sign = -np.sign(u(usqrt(q, qprime, chi), potential) * u(qprime, potential))
    elif Type == 3:
        sign = -np.sign(u(usqrt(q, qprime, chi), potential) *
                        u(usqrt(qprime, qdprime, chip), potential) *
                        Fq(qdprime, ZA_frozen, qvals, deltaq, H_frozen, potential, I_u))

    H_measured[bin_q] += sign
    return H_measured

# ============================ SCATTERING APPROX ============================

@njit
def ScatteringApprox(ZA, potential, qvals, deltaq, H, I_u):
    multiplier = (2.0 * I_u) / (np.pi * ZA)
    running_total = 0.0
    u_0val = u_0(potential)
    for s in range(H.shape[0]):
        running_total += (u(qvals[s], potential) * H[s])
    term = multiplier * running_total
    return u_0val + term

# ============================ ALGORITHM ============================

@njit
def BLDMC(PAA, PAB, PAC, PCA, PCD, PBA, PDC, N, M, q0, deltaq,
          potential, lam, H_approx, ZA_approx):  # PBA = PDC = 1
    '''
    DIAGRAM A -> TYPE 0
    DIAGRAM B -> TYPE 1
    DIAGRAM C -> TYPE 2
    DIAGRAM D -> TYPE 3

    H_approx is the histogram from the previous trial
    ZA_approx is the sum from the that previous trial
    '''
    Type = 0
    Nbins = int(q0 / deltaq)
    qvals = np.arange(Nbins) * deltaq

    # Precompute u weights + CDF once per run (Op 3)
    u_weights = Uweights(qvals, deltaq, potential)
    cdf_u = np.empty(Nbins, dtype=np.float64)
    total_u = BuildCDF(u_weights, cdf_u)

    H_measured = np.zeros(Nbins)
    H_frozen = np.zeros(Nbins)

    I_u = Iu(qvals, deltaq, potential)

    # initial q from |u|
    q = SeedFromCDF(cdf_u, total_u, deltaq)

    qprime = 0.0
    qdprime = 0.0
    chi = 0.0
    chip = 0.0

    ZA_frozen = 1
    DiagramAsum = 0
    DiagramBsum = 0
    DiagramCsum = 0
    DiagramDsum = 0

    scattering_length_array = []

    # arrays for |H_frozen| CDF (Op 2)
    absH = np.empty(Nbins, dtype=np.float64)
    cdf_h = np.empty(Nbins, dtype=np.float64)
    total_h = 0.0
    If_frozen = 0.0

    for i in range(M):
        H_measured[:] = 0.0
        DiagramAsum = 0
        DiagramBsum = 0
        DiagramCsum = 0
        DiagramDsum = 0
        #Type = 0 # maybe not

        if i == 0 and H_approx is not None:
            H_frozen = H_approx
            ZA_frozen = ZA_approx

        # Build CDF for |H_frozen| once per outer iter (Op 2)
        for s in range(Nbins):
            absH[s] = abs(H_frozen[s])
        total_h = BuildCDF(absH, cdf_h)

        # Cache If_frozen once per outer iter (Op 1)
        if ZA_frozen > 0:
            If_frozen = (total_h * I_u) / ZA_frozen
        else:
            If_frozen = 0.0

        for _ in range(N):
            r = np.random.random()

            if Type == 0:
                if r < PAA:
                    q = SeedFromCDF(cdf_u, total_u, deltaq)
                    r2 = np.random.random()
                    if r2 < RAA():
                        Type = 0

                elif r < (PAA + PAB):
                    chi = seedchi()
                    qprime = SeedFromCDF(cdf_h, total_h, deltaq)
                    r2 = np.random.random()
                    if r2 < PXY(RAB_fast(q, qprime, PAB, lam, If_frozen, potential, chi)):
                        Type = 1

                else:  # r < (PAA + PAB + PAC)
                    chi = seedchi()
                    qprime = SeedFromCDF(cdf_u, total_u, deltaq)
                    r2 = np.random.random()
                    if r2 < PXY(RAC(q, qprime, PCA, PAC, lam, qvals, deltaq, potential, chi, I_u)):
                        Type = 2

            elif Type == 1:
                if r < PBA:
                    r2 = np.random.random()
                    # inverse of RAB_fast
                    if r2 < PXY(1.0 / RAB_fast(q, qprime, PAB, lam, If_frozen, potential, chi)):
                        Type = 0

            elif Type == 2:
                if r < PCA:
                    r2 = np.random.random()
                    if r2 < PXY(RCA(q, qprime, PCA, PAC, lam, qvals, deltaq, potential, chi, I_u)):
                        Type = 0
                else:  # r < (PCA + PCD)
                    chip = seedchi()
                    qdprime = SeedFromCDF(cdf_h, total_h, deltaq)
                    r2 = np.random.random()
                    if r2 < PXY(RCD_fast(qprime, qdprime, PCD, If_frozen, potential, chip)):
                        Type = 3

            elif Type == 3:
                if r < PDC:
                    r2 = np.random.random()
                    # inverse of RCD_fast
                    if r2 < PXY(1.0 / RCD_fast(qprime, qdprime, PCD, If_frozen, potential, chip)):
                        Type = 2

            if Type == 0:
                H_measured = HistogramBuilder(0, q, qprime, qdprime, chi,
                                              potential, ZA_frozen, qvals, deltaq,
                                              H_frozen, H_measured, I_u, chip)
                DiagramAsum += 1
            elif Type == 1:
                H_measured = HistogramBuilder(1, q, qprime, qdprime, chi,
                                              potential, ZA_frozen, qvals, deltaq,
                                              H_frozen, H_measured, I_u, chip)
                DiagramBsum += 1
            elif Type == 2:
                H_measured = HistogramBuilder(2, q, qprime, qdprime, chi,
                                              potential, ZA_frozen, qvals, deltaq,
                                              H_frozen, H_measured, I_u, chip)
                DiagramCsum += 1
            elif Type == 3:
                H_measured = HistogramBuilder(3, q, qprime, qdprime, chi,
                                              potential, ZA_frozen, qvals, deltaq,
                                              H_frozen, H_measured, I_u, chip)
                DiagramDsum += 1

        cumulativeZA = True
        if cumulativeZA:
            if DiagramAsum > 0:
                ZA_frozen = DiagramAsum
                H_frozen = H_measured.copy()
            else:
                #print('0 skipped')
                continue
        else: # Doesn't work
            ZA_frozen += DiagramAsum
            H_frozen += H_measured

        approx = ScatteringApprox(ZA_frozen, potential, qvals, deltaq, H_frozen, I_u)
        scattering_length_array.append(approx)

        #print(approx)
        #print(f'DiagramA SUM: {DiagramAsum}')
        #print(f'DiagramB SUM: {DiagramBsum}')  
        #print(f'DiagramC SUM: {DiagramCsum}')
        #print(f'DiagramD SUM: {DiagramDsum}')
        #print()

    return approx, H_frozen, ZA_frozen

def a_analytical(potential):
    '''
    The potential here is defined V = -V_0, the input parameter "potential" wants  V_0, so for -5, input 5
    '''
    kappa = np.sqrt(2 * potential)
    return (1 - np.tan(kappa) / (kappa))

# ============================ RUN ============================

potlist = np.linspace(-0.1, -1.5, 600)
potlistann = np.linspace(-0.1, -1.5, 10000)
num = []
ann = []

H_approx = None
ZA_approx = 0

for i in potlist:
    approx, H_frozen, ZA_frozen = BLDMC(
        PAA=0.2, PAB=0.4, PAC=0.4,
        PCA=0.5, PCD=0.5,
        PBA=1, PDC=1,
        N=1000000000, M=100,
        q0=10, deltaq=0.01,
        potential=i, lam=1,
        H_approx=H_approx, ZA_approx=ZA_approx
    )
    H_approx = H_frozen
    ZA_approx = ZA_frozen
    print(f'Potential: {i} Approx: {approx}')
    num.append(approx)

for i in potlistann:
    ann_scat = a_analytical(potential=abs(i))
    ann.append(ann_scat)

plt.plot(potlist, num,color = 'red', label='Numerical')
plt.plot(potlistann, ann, '--k',label='Analytical')
plt.xlabel('Potential')
plt.ylabel('Scattering Length')
plt.grid(True, which="major", linestyle=":", linewidth=0.7)
plt.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.6)
plt.ylim(-10, 10)

plt.legend()
#plt.savefig('BLDMC1billion.pdf')
#plt.show()
