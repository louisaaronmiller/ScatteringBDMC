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
    #if bin_q < 0 or bin_q >= len(H):
        #return 0.0
    return (D_0 * H[bin_q]) / (S_0 * deltaq) 



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
    acceptance_ratio = ((D0) / (A10 * q0 * abs(u(q,potential,r_star,mass,eps = eps)))) * heaviside(q0,q)
    return acceptance_ratio

def Type1to0(r,A10,q0,q,D0,potential,r_star,mass,eps = epsilon):
    assert 0 <= r <= 1
    if r <= A10:
        if np.random.random() <= Pxy(R10(q,q0,A10,D0,potential,r_star,mass,eps = eps)):
            return True
    return False


# ================================ TYPE 1 ---> 2 ================================

def R12(A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,flag, eps = epsilon,big=K): 
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

def Type1to2(r,A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,flag,eps = epsilon,big=K):  
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

def R21(A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,flag,eps = epsilon,big=K): 
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

def Type2to1(r,A12,A21,q0,q1,q,chi,potential,r_star,mass,H,D0,S0,deltaq,flag,eps = epsilon,big=K):
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

def R2(q,q1,qp,qp1,chi,potential,r_star,mass,H,D0,S0,deltaq,flag,eps = epsilon,big=K): # -------------------------------------------------
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

def variables2(r,q0,A21,lam,q,q1,chi,potential,r_star,mass,H,D0,S0,deltaq,flag,eps = epsilon,big=K):
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
def U(type:int,q,H_measured,H_frozen,deltaq,q0,potential,r_star,mass,S0_frozen,D0,q1,chi,flag,eps=epsilon, big=K):

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
        
        diagram =  - u(q_term,potential,r_star,mass,eps=eps,big=big) * fq1 # Removed 2/pi since it doesn't contribute to sign
        sign = np.sign(diagram)

        H_measured[bin_q] += sign

    return H_measured

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
    kappa = np.sqrt(2 * mass * abs(potential))
    return r_star * (1 - np.tan(kappa * r_star) / (kappa * r_star))


# ================================ ALGORITHM ================================

def DiagMC(N,A10,A12,A21,q0,D0,deltaq,lam,potential,r_star,mass, Niterations = 100, eps = epsilon,big=K):
    '''
    Magic Numbers in the TypeXtoY: 0 corresponds to True or False, 1 corresponds to q,q1
    example: Type0to1[0] = True, Type0to1[1] = q
    '''
    # assert A12 + A21 == 0.1 
    q,q1,chi = seed(q0)
    Nbins = int(q0 / deltaq)
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

        type0sum = 0
        type1sum = 0
        type2sum = 0
        H_measured[:] = 0.0

        if i == 0:
            flag = True # if True, this will use a first approximation of u(q) = f(q)
        else:
            flag = False
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

                elif r < (A10 + A12): 
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
                H_measured = U(1,q,H_measured,H_frozen,deltaq,q0,potential,r_star,mass,type0sum_frozen,D0,q1,chi,flag,eps=epsilon, big=K)
                type1sum += 1
            elif Type == 2:
                H_measured = U(2,q,H_measured,H_frozen,deltaq,q0,potential,r_star,mass,type0sum_frozen,D0,q1,chi,flag,eps=epsilon, big=K)
                type2sum += 1
            else:
                type0sum += 1

        type0sum_frozen = type0sum
        H_frozen = H_measured.copy()
        A = a(type0sum=type0sum_frozen,deltaq=deltaq,D0=D0,qvals =q_vals,H=H_frozen,potential = potential,r_star = r_star,mass = mass,eps=eps,approximation=False)
        scattering_array.append(A)
        print(A)

    total = type0sum + type1sum + type2sum

    return H_frozen,type0sum,type1sum,type2sum,total,Nbins,q_vals,scattering_array


# ================================ RESULTS ================================

H,S0,S1,S2,T,Nbins,q_vals,scattering_array = DiagMC(N=1000000,A10 = 0.2,A12 = 0.4,A21 = 0.4,q0=1.5,D0=1,deltaq=0.001,lam=1.00005,potential=-5,r_star=1,mass=1,Niterations = 20, eps = epsilon, big = K)
scattering_length = a(S0,deltaq=0.001,D0=1,qvals=q_vals,H=H,potential=5,r_star=1,mass=1,eps=epsilon,approximation=False)
scattering_length2 = a(S0,deltaq=0.001,D0=1,qvals=q_vals,H=H,potential=5,r_star=1,mass=1,eps=epsilon,approximation=True)
analytical_scattering_length = a_analytical(potential = 5,r_star = 1, mass = 1)
print(f'Approximation: {scattering_length}')
print(f'Approximation: {scattering_length2}')
print(f'Analytical: {analytical_scattering_length}')

plt.plot(list(range(1, len(scattering_array) + 1)),scattering_array)
plt.axhline(y=analytical_scattering_length, color='r', linestyle='-')
plt.xlabel("q")
plt.ylabel("f(q)")
plt.show()

