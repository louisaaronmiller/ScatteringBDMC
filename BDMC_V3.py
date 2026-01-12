import numpy as np
import matplotlib.pyplot as plt

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
K = 1e5

# ================================ FUNCTIONS ================================

def heaviside(q0,q):
    if q0 < q:
        return 0
    elif q0 >= q:
        return 1

def seed(q0):
    q = np.random.uniform(0,q0)
    q1 = np.random.uniform(0,q0)
    chi = np.random.uniform(-1,1)
    return q,q1,chi

def lambda_seed(lam,q):
    '''
    seeds the value for q' within the range (q/lam,q * lam)
    this is equal to Omega_1 and Omega_2 in the notes
    '''
    q_prime = np.random.uniform(q/lam,q * lam)
    return q_prime

def u_0(potential,r_star,mass):
    return (2/3) * mass * potential * (r_star ** 3)

def u_inf(q,potential,r_star,mass):
    return (-3 * u_0(potential,r_star,mass)) * ((np.cos(q))/(q ** 2))
    
def u(q,potential,r_star,mass,eps = epsilon,big=K):
    if q <= eps:
        return u_0(potential,r_star,mass) * (1 - (q**2)/(2))
    if q >= big:
        return u_inf(q,potential,r_star,mass)
    return (3 * u_0(potential,r_star,mass)) * (np.sin(q) - q * np.cos(q))/(q ** 3)

def Pxy(Rxy):
    '''
    Generalised function for the acceptance probability using acceptance ratios
    x type -> y type
    '''
    return min(1,Rxy)

def get_bin(q,deltaq):
    return int(q//deltaq)

def f(q,D_0,S_0,deltaq,H):
    bin_q = int(q // deltaq)
    return (D_0 * H[bin_q]) / (S_0 * deltaq) # <- probably not correct but next i need to make this at every step (above continue in DiagMC then update the signatures AT THE END)



# ================================ TYPE 0 ---> 1 ================================

def O01(q,q0):
    omega = heaviside(q0,q)/q0
    return omega

def R01(q,q0,A10,D0,potential,r_star,mass,eps = epsilon):
    acceptance_ratio = (A10 * q0 * abs(u(q,potential,r_star,mass,eps = eps))) / (D0)
    return acceptance_ratio

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

def R10(q,q0,A10,D0,potential,r_star,mass,eps = epsilon):
    acceptance_ratio = ((D0) / (A10 * abs(u(q,potential,r_star,mass,eps = eps)))) * heaviside(q0,q)
    return acceptance_ratio

def Type1to0(r,A10,q0,q,D0,potential,r_star,mass,eps = epsilon):
    assert 0 <= r <= 1
    if r <= A10:
        if np.random.random() <= Pxy(R10(q,q0,A10,D0,potential,r_star,mass,eps = eps)):
            return True
    return False


# ================================ TYPE 1 ---> 2 ================================

def R12(A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,eps = epsilon):
    acceptance_ratio = (
        (A21 / A12) *
        (2*q0 * abs(u(np.sqrt(q**2 + q1**2 - 2*q*q1*chi),
                       potential,r_star,mass,eps = eps) * f(q1,D0,S0,deltaq,H))
         ) / (np.pi * abs(u(q,potential,r_star,mass,eps = eps)))
    ) 
    return acceptance_ratio

def Type1to2(r,A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,eps = epsilon): 
    '''
    q is taken from type 1 diagram
    q1 and chi are seeded from (0,q0) and (-1,1) respectively.
    '''
    assert 0 <= r <= 1
    if r <= A12:
        _,q1,chi = seed(q0)
        if np.random.random() <= Pxy(R12(A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,eps = eps)):
            return (True,q,q1,chi)
        else:
            return (False,q,q1,chi)
    return (False,None, None, None)

# ================================ TYPE 1 ---> VARIABLE ================================

def R1(q_prime,q,potential,r_star,mass,eps = epsilon):
    acceptance_ratio = (abs(u(q_prime,potential,r_star,mass,eps = eps)) * q) / (abs(u(q,potential,r_star,mass,eps = eps)) * q_prime)
    return acceptance_ratio

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

def R21(A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,eps = epsilon): # -------------------------------------------------
    acceptance_ratio = heaviside(q0,q1) * (
        (A12 / A21) *
        (np.pi * abs(u(q,potential,r_star,mass,eps = eps))) /
        (2*q0 * abs(u(np.sqrt(q**2 + q1**2 - 2*q*q1*chi),
                       potential,r_star,mass,eps = eps) * f(q1,D0,S0,deltaq,H)))
    )
    return acceptance_ratio

def Type2to1(r,A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,eps = epsilon): # -------------------------------------------------
    '''
    q is taken from type 2 diagram
    q1 and chi are seeded from (0,q0) and (-1,1) respectively.
    '''
    assert 0 <= r <= 1
    if r <= A21:
        if np.random.random() <= Pxy(R21(A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,eps = eps)):
            return (True,q)
    return (False, None)

# ================================ TYPE 2 ---> VARIABLE ================================

def R2(q,q1,qp,qp1,chi,potential,r_star,mass,H,D0,S0,deltaq,eps = epsilon): # -------------------------------------------------
    acceptance_ratio = (
        abs(u(np.sqrt(qp**2 + qp1**2 - 2*qp*qp1*chi),
              potential,r_star,mass,eps = eps) * f(qp1,D0,S0,deltaq,H)) * q*q1
    ) / (
        abs(u(np.sqrt(q**2 + q1**2 - 2*q*q1*chi),
              potential,r_star,mass,eps = eps) * f(q1,D0,S0,deltaq,H)) * qp*qp1
    )
    return acceptance_ratio

def variables2(r,q0,A21,lam,q,q1,chi,potential,r_star,mass,H,D0,S0,deltaq,eps = epsilon):
    '''
    The notes don't have chi_prime or chi_proposal suggesting that you propose chi once and keep using it,
    or you seed chi whenever this is called and use it.
    '''
    assert 0 <= r <= 1
    if r <= (1 - A21):
        qp = lambda_seed(lam,q)
        qp1 = lambda_seed(lam,q1)   
        if np.random.random() <= Pxy(R2(q,q1,qp,qp1,chi,potential,r_star,mass,H,D0,S0,deltaq,eps = eps)):
            #if qp >= q0 or qp1 >= q0: # adding bounds - maybe wrong
            return qp,qp1,chi   # True
        else:
            return q,q1,chi # False
    return q,q1,chi           # False

# ================================ UPDATE FOR TYPE 0 ================================
'''
Nothing happens, the update in type zero is skipped, the only thing you can do is to try to change to type 1
with a addressal probability of 1 and a acceptance probability of Pxy(R01).
'''
# ================================ UPDATE FOR TYPE 1 AND 2 ================================
def U(type:int,q,H,deltaq,q0):
    '''    
    :Type: 0,1,2
    :q: momentum bin
    :H: Histogram
    ''' 
    bin_q = get_bin(q,deltaq)

    if bin_q < 0 or bin_q >= len(H):  #if q < 0 or q >= q0: # adding a bounds - maybe wrong
        return H   # no contribution - this doesn't actally give "no contribution- fix later"

    if type == 1:
        H[bin_q] += 1
    elif type == 2:
        H[bin_q] -= 1
    return H

# ================================ ALGORITHM ================================

def DiagMC(N,A10,A12,A21,q0,D0,deltaq,lam,potential,r_star,mass,eps = epsilon):
    '''
    Magic Numbers in the TypeXtoY: 0 corresponds to True or False, 1 corresponds to q,q1
    example: Type0to1[0] = True, Type0to1[1] = q
    '''
    # assert A12 + A21 == 0.1 
    q,q1,chi = seed(q0)
    Nbins = int(q0 / deltaq)
    H = np.zeros(Nbins * 1000000)   #adds ones to all histogram bins
    Type = 0 # Starting in type 0
    q_vals = np.linspace(0, q0, Nbins)

    type0sum = 0
    type1sum = 0
    type2sum = 0

    for _ in range(N):
        r = np.random.random()

        if _ % 100000 == 0:
            print(q)

        if Type == 0:
            check,q = Type0to1(r,q,q0,D0,potential,r_star,mass,A10,eps = eps)
            if check:
                Type = 1
                type0sum += 1
                continue
            else:
                type0sum += 1
                continue



        if Type == 1:
            if Type1to0(r,A10,q0,q,D0,potential,r_star,mass,eps = eps):
                Type = 0
                type1sum += 1
                continue

            else: 
                check, q_prop,q1_prop,chi_prop = Type1to2(r,A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0=type0sum,deltaq=deltaq,eps = eps)
                if check:
                    q, q1, chi = q_prop, q1_prop, chi_prop
                    Type = 2
                    type1sum += 1
                    continue
                else:
                    if q_prop is not None:
                        q, q1, chi = q_prop, q1_prop, chi_prop
                

            q = variables1(r,q0,A10,A12,lam,q,potential,r_star,mass,eps = eps)
            H = U(1,q,H,deltaq,q0)
            type1sum += 1
            continue



        if Type == 2:
            check,q_prop = Type2to1(r,A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0=type0sum,deltaq=deltaq,eps = eps)
            if check:
                q = q_prop
                Type = 1
                type2sum += 1
                continue

            else:
                q,q1,chi = variables2(r,q0,A21,lam,q,q1,chi,potential,r_star,mass,H,D0,S0=type0sum,deltaq=deltaq,eps = eps)
                H = U(2,q,H,deltaq,q0)
                type2sum += 1
                continue
    total = type0sum + type1sum + type2sum
    return H,type0sum,type1sum,type2sum,total,Nbins,q_vals

# ================================ SCATTERING LENGTH ================================

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



# ================================ SCATTERING LENGTH APPROXIMATION ================================



# ================================ RESULTS ================================

H,S0,S1,S2,T,Nbins,q_vals = DiagMC(N=1000000,A10 = 0.01,A12 = 0.05,A21 = 0.05,q0=1.5,D0=1,deltaq=0.001,lam=1.5,potential=6,r_star=1,mass=1,eps = epsilon)
scattering_length = a(S0,deltaq=0.001,D0=1,qvals=q_vals,H=H,potential=5,r_star=1,mass=1,eps=epsilon,approximation=False)
scattering_length2 = a(S0,deltaq=0.001,D0=1,qvals=q_vals,H=H,potential=5,r_star=1,mass=1,eps=epsilon,approximation=True)
analytical_scattering_length = a_analytical(potential = 5,r_star = 1, mass = 1)
print(f'{scattering_length}')
print(f'{analytical_scattering_length}')
#plt.hist(H, bins=Nbins, density=True, label='Scattering BDMC')
plt.plot(q_vals, H / T)   # T = total MC steps
plt.xlabel("q")
plt.ylabel("f(q)")
plt.show()


# When this seeds values i.e. q,q1; it does it with np.random.uniform()
