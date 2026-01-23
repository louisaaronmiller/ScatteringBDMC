# -------------------- Modules and Packages--------------------

# import scipy.constants as sc
import math as math
from numba import njit
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.special import spherical_jn, spherical_yn

# -------------------- Constants and Parameters--------------------

R = 1  # radius of the potential 1e-15
V_0 = -5  # -10e6 * 1.6e-19
E = 13  # should eventually become an array
rmax = 20
# taking hbar^2 / 2m == 1

# -------------------- Potential, Energy, and Schrodinger terms --------------------
@njit
def V(r,R=R,V_0=V_0):
    if r < R:
        return V_0
    else:

        return 0

@njit
def k(E, r,R=R,hbar = 1):
    if r > R:
        return math.sqrt(2 *E)/hbar
    else:
        return math.sqrt(E - V_0)
    

@njit
def k_func(m, E, l, r, R, V_0, hbar = 1.0):
    # k_eff(r) = 2mE/hbar^2 - l(l+1)/r^2 - 2mV(r)/hbar^2
    if r == 0.0:
        return 0.0
    inv_hbar2 = 1.0 / (hbar * hbar)
    return (2.0*m*E)*inv_hbar2 - (l*(l+1.0))/(r*r) - (2.0*m*V(r, R, V_0))*inv_hbar2

@njit
def Numerov(l, E, rmax, R, h, V_0, m=1.0, hbar=1.0):
    rvals = np.arange(0.0, rmax, h)
    n = len(rvals)
    y = np.zeros(n)

    y[0] = 0.0
    y[1] = h ** (l + 1)

    h2 = h * h
    c = h2 / 12.0

    for i in range(1, n - 1):
        km1 = k_func(m, E, l, rvals[i - 1], R, V_0, hbar)
        kn  = k_func(m, E, l, rvals[i],     R, V_0, hbar)
        kp1 = k_func(m, E, l, rvals[i + 1], R, V_0, hbar)

        y[i + 1] = (2.0 * (1.0 - 5.0*c*kn) * y[i] - (1.0 + c*km1) * y[i - 1]) / (1.0 + c*kp1)

    return y, rvals



# -------------------- Functions for getting r and u values starting from outside the potential --------------------

def outside_valsold(rvals, uvals,R=R):
    """
    returns a tuple where the u and r vals start just when the potential "turns off"
    """
    rvals_dict = dict(enumerate(rvals))
    for key, value in rvals_dict.items():
        if V(value,R) != 0:
            continue
        else:
            index = key
            break
    r = rvals[index:]
    u = uvals[index:]
    return u, r

@njit
def outside_vals(rvals, uvals, R=R):
    index = 0
    n = len(rvals)

    for i in range(n):
        if V(rvals[i], R) == 0.0:
            index = i
            break

    return uvals[index:], rvals[index:]


def r_1halfr_2old(r, u, E, max_points=2):
    """
    Finds r and u values at successive extrema (half-wavelength separation).
    Works by finding local maxima/minima in u(r).
    """

    u = np.array(u)
    r = np.array(r)

    # Finding indices of local maxima and minima
    extrema_idx = (argrelextrema(u, np.greater)[0].tolist() + argrelextrema(u, np.less)[0].tolist())
    extrema_idx.sort()

    extrema_idx = extrema_idx[:max_points]

    r_aug = r[extrema_idx]
    u_aug = u[extrema_idx]

    return r_aug, u_aug

@njit
def r_1halfr_2(r, u,E, max_points=2):
    n = len(u)

    r_out = np.empty(max_points)
    u_out = np.empty(max_points)
    count = 0

    for i in range(1, n - 1):
        if (u[i] > u[i-1] and u[i] > u[i+1]) or (u[i] < u[i-1] and u[i] < u[i+1]):
            r_out[count] = r[i]
            u_out[count] = u[i]
            count += 1
            if count == max_points:
                break

    return r_out[:count], u_out[:count]


# -------------------- Functions for phase shifts (delta) and total cross section (sigma) --------------------
@njit
def K(rvals, uvals):
    """
    rvals here are starting from when there is no potential r > R

    The l value the uvals possesses will determine the l value/subscript
    that the phase shift delta_l will have.
    """
    K_array = []
    for i in range(len(rvals) - 1):
        K = (rvals[i] * uvals[i + 1]) / (rvals[i + 1] * uvals[i])
        K_array.append(K)
    return K_array


def delta_lOLD(l, rvals, kvals, E):
    """
    similar to K, rvals here are starting from when there is no potential r > R
    """

    deltavals = []
    k_0 = math.sqrt(2*E)

    for i in range(len(rvals) - 1):
        j_l_i = spherical_jn(l, k_0 * rvals[i])
        n_l_i = spherical_yn(l, k_0 * rvals[i])

        j_l_ip1 = spherical_jn(l, k_0 * rvals[i + 1])
        n_l_ip1 = spherical_yn(l, k_0 * rvals[i + 1])

        numerator = kvals[i] * j_l_i - j_l_ip1
        denominator = kvals[i] * n_l_i - n_l_ip1

        delta_i = np.arctan(numerator / denominator)

        deltavals.append(delta_i)
    return deltavals

@njit
def delta_l(l, rvals, kvals, E):
    """
    Numba-compatible version of your delta_l for l=0 only.
    Same signature, returns a Python list of delta values.

    Uses:
      j0(x) = sin(x)/x
      n0(x) = -cos(x)/x
    """
    deltavals = []
    k_0 = math.sqrt(2.0 * E)

    for i in range(len(rvals) - 1):
        x_i   = k_0 * rvals[i]
        x_ip1 = k_0 * rvals[i + 1]

        # avoiding /zero 
        if x_i == 0.0 or x_ip1 == 0.0:
            deltavals.append(0.0)
            continue

        j0_i   = math.sin(x_i)   / x_i
        n0_i   = -math.cos(x_i)  / x_i
        j0_ip1 = math.sin(x_ip1) / x_ip1
        n0_ip1 = -math.cos(x_ip1)/ x_ip1

        numerator   = kvals[i] * j0_i - j0_ip1
        denominator = kvals[i] * n0_i - n0_ip1

        delta_i = np.arctan(numerator / denominator)

        deltavals.append(delta_i)

    return deltavals

@njit
def sigma(l, delta,E):
    k = math.sqrt(2*E)
    if l == 0:
        sigma_tot = ((4 * np.pi) / (k**2)) * (np.sin(delta)) ** 2
        return sigma_tot
    else:
        return 0  # Set up later, should be a sum from l=0 to infinity, going to some l that stops when terms after aren't as 'strong'

# -------------------- Delta, Sigma, Energies, Momenta simulation --------------------

def PhaseforEnergy(l: int,E_min = 0,E_max = 30,rmax = rmax,R=R, h = 0.001,V_0=V_0,flag = True):
    E = np.arange(E_min, E_max, 0.1)
    if flag is False:
        E = np.linspace(E_min,E_max,500)
    delta = []
    k = []
    sigmas = []
    for i in E:
        k.append(np.sqrt(2 *i)) #CHANGED BECAUSE OF HBAR^2/2M = 1 CHANGE

        u, r = Numerov(l, i, rmax, R, h,V_0)
        u_aug, r_aug = outside_vals(r, u,R)
        r_new, u_new = r_1halfr_2(r_aug, u_aug, i)

        phaseshift = delta_l(l, r_new, K(r_new, u_new), i)
        # print(f"phase shifts: {phaseshift}")
        delta.append(phaseshift[-1])

        sig = sigma(l,phaseshift[-1],i)
        sigmas.append(sig)
    return delta, E, k, sigmas

# -------------------- Analytical Functions --------------------

def analytical_delta0(E,V_0=V_0,a=R):
    '''
        Calculates analytical phase shift values for l=0

        a = radius of the spherical potential

        Taken from B.H.B & C.J.J Quantum mechanics 2nd Edition,
        they define K as k**2 + V_0 and k^2 as 2mE/hbar**2.
        Here k is defined as sqrt(E) and K as sqrt(E+V_0),
        assuming a attractive potential.
    '''
    k = math.sqrt(2*E) #CHANGED BECAUSE OF HBAR^2/2M = 1 CHANGE
    K = math.sqrt(E-V_0) # Because i've defined V_0 has negative the minus signs cancels eachother out


    numerator = k * np.tan(K*a) - K * np.tan(k*a)
    denominator = K + (k * np.tan(k*a) * np.tan(K*a))
    delta = np.arctan(numerator/denominator) #math.atan2

    return delta

def analytical_deltal(E,l,V_0=V_0,a=R):
    '''
        The same function as analytical_delta0, but for l>0
    '''
    if V_0 < 0:
        k = math.sqrt(2* E) #CHANGED BECAUSE OF HBAR^2/2M = 1 CHANGE
        K = math.sqrt(E-V_0)
    else:
        k = math.sqrt(2 *E) #CHANGED BECAUSE OF HBAR^2/2M = 1 CHANGE
        K = math.sqrt(E +V_0)
    # -------- k --------

    j_l_k = spherical_jn(l, k * a)
    n_l_k = spherical_yn(l, k * a)

    j_l_k_prime = spherical_jn(l, k * a, derivative=True)
    n_l_k_prime = spherical_yn(l, k * a, derivative=True)

    # -------- K --------

    j_l_K = spherical_jn(l, K * a)
    #n_l_K = spherical_yn(l, K * a)

    j_l_K_prime = spherical_jn(l, K * a, derivative=True)
    #n_l_K_prime = spherical_yn(l, K * a, derivative=True)

    # -------- Calculation --------

    numerator = (k * j_l_k_prime * j_l_K) - (K * j_l_k * j_l_K_prime)
    denominator = (k * n_l_k_prime * j_l_K) - (K * n_l_k * j_l_K_prime)

    delta = np.arctan(numerator/denominator) #math.atan2

    return delta

def analytical_sigma0(delta,E):
    k = math.sqrt(2 *E)

    sig = ((4 * np.pi) / (k **2)) * (np.sin(delta))**2
    
    return sig
# -------------------- Resonance (Bound states) --------------------

def delta_res(l, rvals, kvals, E):
    """
    creates delta values for 
    """

    deltavals = []
    k_0 = math.sqrt(E)

    for i in range(len(rvals) - 1):
        j_l_i = spherical_jn(l, k_0 * rvals[i])
        n_l_i = spherical_yn(l, k_0 * rvals[i])

        j_l_ip1 = spherical_jn(l, k_0 * rvals[i + 1])
        n_l_ip1 = spherical_yn(l, k_0 * rvals[i + 1])

        numerator = kvals[i] * j_l_i - j_l_ip1
        denominator = kvals[i] * n_l_i - n_l_ip1

        delta_i = np.arctan(numerator / denominator)

        deltavals.append(delta_i)
    return deltavals

@njit
def ScatterLength(l: int, E=1e-3, rmax=rmax, R=R, h=0.0001, flag=True, E_min=0, E_max=1):
    V_0 = np.linspace(-1.5, 0.0, 1000)
    nV = len(V_0)

    delta  = np.empty(nV)
    k_arr  = np.empty(nV)
    sigmas = np.empty(nV)

    for idx in range(nV):
        pot = V_0[idx]

        k_arr[idx] = math.sqrt(2.0 * E)

        u, r = Numerov(l, E, rmax, R, h, pot)
        u_aug, r_aug = outside_vals(r, u, R)
        r_new, u_new = r_1halfr_2(r_aug, u_aug, E)

        kvals = K(r_new, u_new)
        phaseshift = delta_l(l, r_new, kvals, E)

        delta[idx] = phaseshift[-1]
        sigmas[idx] = sigma(l, delta[idx], E)

    scattering_length = -delta / k_arr
    V_0abs = np.abs(V_0)

    return delta, V_0, k_arr, sigmas, scattering_length, V_0abs

@njit
def AnSL(a):
    V_0 = np.linspace(1.5,0,100) ##### MATCH----------------------
    SLarr = []
    for i in V_0:
        gamma = (np.sqrt(2)*i**0.5) * a
        SL = (1 - (np.tan(gamma))/(gamma)) * a
        SLarr.append(SL)
    return V_0,SLarr

def wavefunctionunshifted(l: int, E: float, rmax, R, V_0, h = 0.01):
    u, r =  Numerov(l=l, E=E,rmax = rmax,R=R, h = 0.01,V_0=V_0)
    return u,r


# -----------------------------------------------------------------------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from numba import njit

'''
x,y are integers that represent the type you are in.

Axy: Addressal probability to transfer from type x to type y.
Pxy: Acceptance probability of transfering from type x to type y.
Rxy: Acceptance ratio, that is used in the acceptance probability.
Oxy: Function that you seed with, O is for Omega.

U: update -> works for all types (type is in function signature, thus the function is called with a type)

Rx: this refers to the variables-x acceptance ratio, you can use this acceptance ratio with Pxy. 

C: normalisation constant
D0: normalisation constant related to type 0

q,q1 are seeded from the uniform interval (0,q0)  ------ This affects the acceptance ratios (simplifying them)
chi is seeded from the unfirom interval (-1,1)    ------ This affects the acceptance ratios (simplifying them)
'''

# ================================ CONSTANTS ================================

epsilon = 1e-5
K_big = 1e5

# ================================ FUNCTIONS ================================
@njit
def heaviside(q0,q):
    if q0 < q:
        return 0
    elif q0 > q:
        return 1
@njit
def seed(q0):
    q = np.random.uniform(0,q0)
    q1 = np.random.uniform(0,q0)
    chi = np.random.uniform(-1,1)
    return q,q1,chi
@njit
def lambda_seed(lam,q):
    '''
    seeds the value for q' within the range (q/lam,q * lam)
    this is equal to Omega_1 and Omega_2 in the notes
    '''
    q_prime = np.random.uniform(q/lam,q * lam)
    return q_prime
@njit
def u_0(potential,r_star,mass):
    return (2/3) * mass * potential * (r_star ** 3)
@njit
def u_inf(q,potential,r_star,mass):
    return (-3 * u_0(potential,r_star,mass)) * ((np.cos(q))/(q ** 2))
@njit
def u(q,potential,r_star,mass,eps = epsilon,big=K_big):
    if q <= eps:
        return u_0(potential,r_star,mass) * (1 - (q**2)/(2))
    if q >= big:
        return u_inf(q,potential,r_star,mass)
    return (3 * u_0(potential,r_star,mass)) * (np.sin(q) - q * np.cos(q))/(q ** 3)
@njit
def Pxy(Rxy):
    '''
    Generalised function for the acceptance probability using acceptance ratios
    x type -> y type
    '''
    return min(1,Rxy)
@njit
def get_bin(q,deltaq):
    return int(q//deltaq)
@njit
def f(q,D_0,S_0,deltaq,H):
    bin_q = int(q // deltaq)
    if bin_q < 0 or bin_q >= len(H):
        return 0.0
    return (D_0 * H[bin_q]) / (S_0 * deltaq) 



# ================================ TYPE 0 ---> 1 ================================

def O01(q,q0):
    omega = heaviside(q0,q)/q0
    return omega
@njit
def R01(q,q0,A10,D0,potential,r_star,mass,eps = epsilon):
    acceptance_ratio = (A10 * q0 * abs(u(q,potential,r_star,mass,eps = eps))) / (D0)
    return acceptance_ratio
@njit
def Type0to1(r,q,q0,D0,potential,r_star,mass,A10,eps = epsilon):
    '''
    Returns True if you accept type swap
    Returns False if you reject type swap

    Addressal probability is equal to one. (A01= 1)
    '''
    assert 0 <= r <= 1
    q,_,_ = seed(q0)
    if r <= Pxy(R01(q,q0,A10,D0,potential,r_star,mass,eps = eps)):
        return (True,q)
    else:
        return (False,q)

# ================================ TYPE 1 ---> 0 ================================
@njit
def R10(q,q0,A10,D0,potential,r_star,mass,eps = epsilon):
    acceptance_ratio = ((D0) / (A10 * q0 * abs(u(q,potential,r_star,mass,eps = eps)))) * heaviside(q0,q)
    return acceptance_ratio
@njit
def Type1to0(r,A10,q0,q,D0,potential,r_star,mass,eps = epsilon):
    assert 0 <= r <= 1
    if r <= A10:
        if np.random.random() <= Pxy(R10(q,q0,A10,D0,potential,r_star,mass,eps = eps)):
            return True
    return False


# ================================ TYPE 1 ---> 2 ================================
@njit
def R12(A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,flag, eps = epsilon,big=K_big): 
    if flag == True:
        acceptance_ratio = (
        (A21 / A12) *
        (2*q0 * abs(u(np.sqrt(q**2 + q1**2 - 2*q*q1*chi),
                       potential,r_star,mass,eps = eps) * u(q1,potential,r_star,mass,eps = eps,big=big))
         ) / (np.pi * abs(u(q,potential,r_star,mass,eps = eps)))
    ) 
    else:
        acceptance_ratio = (
        (A21 / A12) *
        (2*q0 * abs(u(np.sqrt(q**2 + q1**2 - 2*q*q1*chi),
                       potential,r_star,mass,eps = eps) * f(q1,D0,S0,deltaq,H))
         ) / (np.pi * abs(u(q,potential,r_star,mass,eps = eps)))
    ) 
    return acceptance_ratio
@njit
def Type1to2(r,A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,flag,eps = epsilon,big=K_big):  
    '''
    q is taken from type 1 diagram
    q1 and chi are seeded from (0,q0) and (-1,1) respectively.
    '''
    assert 0 <= r <= 1
    if r <= A12:
        _,q1,chi = seed(q0)
        if np.random.random() <= Pxy(R12(A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,flag,eps = eps,big=big)):
            return (True,q,q1,chi)
        else:
            return (False,q,q1,chi)
    return (False,None, None, None)

# ================================ TYPE 1 ---> VARIABLE ================================
@njit
def R1(q_prime,q,potential,r_star,mass,eps = epsilon):
    acceptance_ratio = (abs(u(q_prime,potential,r_star,mass,eps = eps)) * q) / (abs(u(q,potential,r_star,mass,eps = eps)) * q_prime)
    return acceptance_ratio
@njit
def variables1(r,q0,A10,A12,lam,q,potential,r_star,mass,eps = epsilon):
    assert 0 <= r <= 1
    if r <= (1 - A10 - A12):
        q_prime = lambda_seed(lam,q)
        if np.random.random() <= Pxy(R1(q_prime,q,potential,r_star,mass,eps = eps)):
            #if q_prime >= q0: # adding bounds - maybe wrong - if continuing with this, swap return
            return q_prime # True
        else:
            return q # False
    return q               # False

# ================================ TYPE 2 ---> 1 ================================
@njit
def R21(A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,flag,eps = epsilon,big=K_big): 
    if flag:
        acceptance_ratio = heaviside(q0,q1) * (
        (A12 / A21) *
        (np.pi * abs(u(q,potential,r_star,mass,eps = eps))) /
        (2*q0 * abs(u(np.sqrt(q**2 + q1**2 - 2*q*q1*chi),
                       potential,r_star,mass,eps = eps) * u(q1,potential,r_star,mass,eps = eps,big=big)))
    )
    else:
        acceptance_ratio = heaviside(q0,q1) * (
            (A12 / A21) *
            (np.pi * abs(u(q,potential,r_star,mass,eps = eps))) /
            (2*q0 * abs(u(np.sqrt(q**2 + q1**2 - 2*q*q1*chi),
                        potential,r_star,mass,eps = eps) * f(q1,D0,S0,deltaq,H)))
        )
    return acceptance_ratio

@njit
def Type2to1(r,A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,flag,eps = epsilon,big=K_big):
    '''
    q is taken from type 2 diagram
    q1 and chi are seeded from (0,q0) and (-1,1) respectively.
    '''
    assert 0 <= r <= 1
    if r <= A21:
        if np.random.random() <= Pxy(R21(A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,flag,eps = eps,big=big)):
            return (True,q)
    return (False, None)

# ================================ TYPE 2 ---> VARIABLE ================================
@njit
def R2(q,q1,qp,qp1,chi,potential,r_star,mass,H,D0,S0,deltaq,flag,eps = epsilon,big=K_big): # -------------------------------------------------
    if flag:
        
        acceptance_ratio = (
        abs(u(np.sqrt(qp**2 + qp1**2 - 2*qp*qp1*chi),
              potential,r_star,mass,eps = eps) * u(qp1,potential,r_star,mass,eps = eps,big=big)) * q*q1
    ) / (
        abs(u(np.sqrt(q**2 + q1**2 - 2*q*q1*chi),
              potential,r_star,mass,eps = eps) * u(q1,potential,r_star,mass,eps = eps,big=big)) * qp*qp1
    )

    else:

        acceptance_ratio = (
            abs(u(np.sqrt(qp**2 + qp1**2 - 2*qp*qp1*chi),
                potential,r_star,mass,eps = eps) * f(qp1,D0,S0,deltaq,H)) * q*q1
        ) / (
            abs(u(np.sqrt(q**2 + q1**2 - 2*q*q1*chi),
                potential,r_star,mass,eps = eps) * f(q1,D0,S0,deltaq,H)) * qp*qp1
        )
    return acceptance_ratio

@njit
def variables2(r,q0,A21,lam,q,q1,chi,potential,r_star,mass,H,D0,S0,deltaq,flag,eps = epsilon,big=K_big):
    '''
    The notes don't have chi_prime or chi_proposal suggesting that you propose chi once and keep using it,
    or you seed chi whenever this is called and use it.
    '''
    assert 0 <= r <= 1
    if r <= (1 - A21):
        qp = lambda_seed(lam,q)
        qp1 = lambda_seed(lam,q1)   
        if np.random.random() <= Pxy(R2(q,q1,qp,qp1,chi,potential,r_star,mass,H,D0,S0,deltaq,flag,eps = eps,big=big)):
            #if qp >= q0 or qp1 >= q0: # adding bounds - maybe wrong
            return qp,qp1,chi # True
        else:
            return q,q1,chi # False
    return q,q1,chi           # False

# ================================ UPDATE FOR TYPE 0 ================================
'''
Nothing happens, the update in type zero is skipped, the only thing you can do is to try to change to type 1
with a addressal probability of 1 and a acceptance probability of Pxy(R01).
'''
# ================================ UPDATE FOR TYPE 1 AND 2 ================================
@njit
def U(type:int,q,H_measured,H_frozen,deltaq,q0,potential,r_star,mass,S0_frozen,D0,q1,chi,flag,eps=epsilon, big=K_big):

    bin_q = get_bin(q,deltaq)

    if bin_q < 0 or bin_q >= len(H_measured):  #if q < 0 or q >= q0: # adding a bounds - maybe wrong
        return H_measured   # no contribution - this doesn't actally give "no contribution- fix later"
    

    if type == 1:
        H_measured[bin_q] += np.sign(u(q,potential,r_star,mass,eps=eps,big=big))
    elif type == 2:
        q_term = np.sqrt(q ** 2 + q1 ** 2 - (2 * q * q1 * chi))
        if flag:
            fq1 = u(q1,potential,r_star,mass,eps=eps,big=big)
        else:
            fq1 = (D0 * H_frozen[get_bin(q1,deltaq)]) / (S0_frozen * deltaq)
            if fq1 == 0.0:
                return H_measured 
        
        diagram =  - u(q_term,potential,r_star,mass,eps=eps,big=big) * fq1 # Removed 2/pi since it doesn't contribute to sign
        sign = np.sign(diagram)

        H_measured[bin_q] += sign

    return H_measured

# ================================ SCATTERING LENGTH ================================
@njit
def a(type0sum,deltaq,D0,qvals,H,potential,r_star,mass,eps=epsilon,approximation=False):
    if approximation:
        return (D0 * H[0])/(type0sum * (deltaq))
    else:
        u_0 = u(0,potential,r_star,mass,eps=eps)
        running_total = 0
        for s in range(len(H)):
            running_total += u(qvals[s],potential,r_star,mass,eps=eps) * H[s]
        scattering_length = u_0 - ((2 * D0) / (np.pi * type0sum) * running_total)
        return scattering_length
    

def a_analytical(potential,r_star,mass):
    '''
    The potential here is defined V = -V_0, the input parameter "potential" wants  V_0, so for -5, input 5
    '''
    kappa = np.sqrt(2 * mass * potential)
    return r_star * (1 - np.tan(kappa * r_star) / (kappa * r_star))


# ================================ ALGORITHM ================================
@njit
def DiagMC(N,A10,A12,A21,q0,D0,deltaq,lam,potential,r_star,mass, Niterations = 100,use_last_block = None, use_last_t0sum = 0, eps = epsilon,big=K_big):
    '''
    Set use_last_block = None to NOT use a starting histogram approximation from a previous calculation,
    otherwise set use_last_block to a histogram that you want to sample from. This is accompanied with
    use_last_t0sum, which is type0sum_frozen that corresponds to the use_last_block sample
    '''
    # assert A12 + A21 == 0.1 
    q,q1,chi = seed(q0)
    Nbins = int(q0/deltaq)
    H_measured = np.zeros(Nbins) # measured during DiagMC then used for the next iteration (frozen)
    H_frozen = np.zeros(Nbins) # used within acceptance ratios


    Type = 0 # Starting in type 0
    q_vals = np.linspace(0, q0, Nbins)

    

    type0sum = 0
    type1sum = 0
    type2sum = 0
    type0sum_frozen = 0


    scattering_array = []

    for i in range(Niterations):

        Type = 0
        type0sum = 0
        type1sum = 0
        type2sum = 0
        H_measured[:] = 0.0

        if use_last_block is None:
            if i == 0:
                flag = True # if True, this will use a first approximation of u(q) = f(q)
            else:
                flag = False
        else:
            H_frozen = use_last_block
            type0sum_frozen = use_last_t0sum



        for _ in range(N):
            r = np.random.random()


            if Type == 0:
                check,q_prop= Type0to1(r,q,q0,D0,potential,r_star,mass,A10,eps = eps)
                q = q_prop
                if check:
                    Type = 1


            elif Type == 1:
                if r < A10:
                

                    if Type1to0(0,A10,q0,q,D0,potential,r_star,mass,eps = eps):
                        Type = 0

                elif r < (A12): 
                    check, q_prop,q1_prop,chi_prop = Type1to2(0,A12,A21,q0,q1,q,chi,potential,r_star,mass,H_frozen,D0,S0=type0sum_frozen,deltaq=deltaq,flag=flag,eps = eps,big=big)
                    if q_prop is not None:
                            q, q1, chi = q_prop, q1_prop, chi_prop
                    if check:
                        Type = 2
                else:

                    q = variables1(0,q0,A10,A12,lam,q,potential,r_star,mass,eps = eps)


            elif Type == 2:
                if r < A21:
                    check,q_prop = Type2to1(0,A12,A21,q0,q1,q,chi,potential,r_star,mass,H_frozen,D0,S0=type0sum_frozen,deltaq=deltaq,flag=flag,eps = eps,big=big) 
                    if check:
                        q = q_prop
                        Type = 1

                else:
                    q,q1,chi = variables2(0,q0,A21,lam,q,q1,chi,potential,r_star,mass,H_frozen,D0,S0=type0sum_frozen,deltaq=deltaq,flag=flag,eps = eps,big=big) # ------------------------------------------


            if Type == 1:
                H_measured = U(1,q,H_measured,H_frozen,deltaq,q0,potential,r_star,mass,type0sum_frozen,D0,q1,chi,flag,eps=epsilon, big=big)
                type1sum += 1
            elif Type == 2:
                H_measured = U(2,q,H_measured,H_frozen,deltaq,q0,potential,r_star,mass,type0sum_frozen,D0,q1,chi,flag,eps=epsilon, big=big)
                type2sum += 1
            else:
                type0sum += 1

        type0sum_frozen = type0sum
        H_frozen = H_measured.copy()
        A = a(type0sum=type0sum_frozen,deltaq=deltaq,D0=D0,qvals =q_vals,H=H_frozen,potential = potential,r_star = r_star,mass = mass,eps=eps,approximation=False)
        scattering_array.append(A)
        print(A)
        print(f'Type-0 SUM: {type0sum}')
        print(f'Type-1 SUM: {type1sum}')
        print(f'Type-2 SUM: {type2sum}')
        print()

    total = type0sum + type1sum + type2sum

    return H_frozen,type0sum,type1sum,type2sum,total,Nbins,q_vals,scattering_array,type0sum_frozen


# ================================ RESULTS ================================
potlist = np.linspace(0.0, -1.5, 100)
potlistann = np.linspace(0,-1.5,1000)
scatter_length_ann = []
scatter_length_num = []
scat3 = []

use_last_block = None
use_last_t0sum = 0
for Zpot in potlist:
    H,S0,S1,S2,T,Nbins,q_vals,scattering_array,TzeroSum = DiagMC(N=10000000,A10 = 0.2,A12 = 0.4,A21 = 0.4,q0=20,D0=1,deltaq=0.0001,lam=1.5,potential=Zpot,r_star=1,mass=1,Niterations = 20, use_last_block = use_last_block, use_last_t0sum = use_last_t0sum, eps = epsilon, big = K_big)
    scatter_length_num.append(scattering_array[-1])

    use_last_block = H
    use_last_t0sum = TzeroSum

for i in potlistann:
    ann_scat = a_analytical(potential = abs(i),r_star = 1, mass = 1)
    scatter_length_ann.append(ann_scat)
#delta, V_0, k, sigmas, scattering_length,V_0abs = ScatterLength(l=0,E=1e-4,rmax=600,R=1)
#V_0an, SLan = AnSL(1)


#scattering_length = a(S0,deltaq=0.001,D0=1,qvals=q_vals,H=H,potential=5,r_star=1,mass=1,eps=epsilon,approximation=False)
#scattering_length2 = a(S0,deltaq=0.001,D0=1,qvals=q_vals,H=H,potential=5,r_star=1,mass=1,eps=epsilon,approximation=True)
analytical_scattering_length = a_analytical(potential = 0.5,r_star = 1, mass = 1)

#print(f'Approximation: {scattering_length}')
#print(f'Approximation: {scattering_length2}')
print(f'Analytical: {analytical_scattering_length}')


fig, ax = plt.subplots()
#ax.plot(list(range(1, len(scattering_array) + 1)),scattering_array)
#plt.axhline(y=analytical_scattering_length, color='r', linestyle='-',label = 'Analytical')
#plt.xlabel("Number of Iterations")
#plt.ylabel("Scattering Length")


delta, V_0, k, sigmas, scattering_length,V_0abs = ScatterLength(l=0,E=1e-4,rmax=600
                                                         ,R=1)

plt.plot(V_0 ,scattering_length,color = 'k',label = 'Numerov')

ax.set_ylim(-10, 10)

ax.scatter(potlist ,scatter_length_num,color = 'c',s=35,label = 'BDMC')
ax.plot(potlistann,scatter_length_ann,'--r',label = 'Analytical')
#ax.plot(potlist,scatter_length,color = 'r', label = 'BDMC')
plt.xlabel('Potential $V_0$')
plt.ylabel('Scattering Length')




# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Keep bottom and left spines
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

# Optional: make ticks point outward
ax.tick_params(direction='out')
ax.grid()
plt.legend()
plt.show()
