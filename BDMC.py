import numpy as np

'''
x,y are integers that represent the type you are in.

Axy: Addressal probability to transfer from type x to type y.
Pxy: Acceptance probability of transfering from type x to type y.
Rxy: Acceptance ratio, that is used in the acceptance probability.
Oxy: Function that you seed with, O is for Omega.

Ux: Update for the type x

Rx: this refers the variables-x acceptance ratio, you can use this acceptance ratio with Pxy. 

C: normalisation constant
D0: normalisation constant related to type 0

q,q1 are seeded from the uniform interval (0,q0)  ------ This affects the acceptance ratios (simplifying them)
chi is seeded from the unfirom interval (-1,1)    ------ This affects the acceptance ratios (simplifying them)
'''

def heaviside(q0,q):
    if q0 < q:
        return 0
    elif q0 >= q:
        return 1

def seed(q0):
    q= np.random.uniform(0,q0)
    chi = np.random.uniform(-1,1)
    return q,chi

def lambda_seed(lam,q):
    '''
    seeds the value for q' within the range (q/lam,q * lam)
    this is equal to Omega_1 and Omega_2 in the notes
    '''
    q_prime = np.random.uniform(q/lam,q * lam)
    return q_prime

def u(q):
    return 1

def f(q1):
    return 1

def Pxy(Rxy):
    '''
    Generalised function for the acceptance probability using acceptance ratios
    x type -> y type
    '''
    return min(1,Rxy)

# ================================ TYPE 0 ---> 1 ================================

def O01(q,q0):
    omega = heaviside(q0,q)/q0
    return omega

def R01(q,q0,A01,D0):
    acceptance_ratio = (A01 * q0 * abs(u(q))) / (D0)
    return acceptance_ratio

def Type0to1(q,q0,D0,A01=1):
    '''
    Returns True if you accept type swap
    Returns False if you reject type swap

    Addressal probability is equal to one. (A01= 1)
    '''
    if np.random.random() <= Pxy(R01(q,q0,A01,D0)):
        return True
    else:
        return False

# ================================ TYPE 1 ---> 0 ================================

def R10(q,q0,A10,D0):
    acceptance_ratio = ((D0) / (A10 * abs(u(q)))) * heaviside(q0,q)
    return acceptance_ratio

def Type1to0(A10,q0,q,D0):
    if np.random.random() <= A10:
        if np.random.random() <= Pxy(R10(q,q0,A10,D0)):
            return True
    return False


# ================================ TYPE 1 ---> 2 ================================

def O12(q0,q1): # notes say dependence on q and chi are here, but they're not?
    heaviside(q0,q1) / (2 * q0)

def R12(A12,A21,q0,q1,q,chi):
    acceptance_ratio = ( (A21) / (A12) ) * ( (2*q0*abs(u(np.sqrt((q**2) + (q1**2) - 2*q*q1*chi)) * f(q1))) / (np.pi * abs(u(q))) )
    return acceptance_ratio

def Type1to2(A12,A21,q0,q1,q,chi):
    '''
    q is taken from type 1 diagram
    q1 and chi are seeded from (0,q0) and (-1,1) respectively.
    '''
    if np.random.random() <= A12:
        if np.random.random() <= Pxy(R12(A12,A21,q0,q1,q,chi)):
            return True
    return False

# ================================ TYPE 1 ---> VARIABLE ================================

def R1(q_prime,q):
    acceptance_ratio = (abs(u(q_prime)) * q) / (abs(u(q)) * q_prime)
    return acceptance_ratio

def variables1(A10,A12,lam,q):
    if np.random.random() <= (1 - A10 - A12):
        q_prime = lambda_seed(lam,q)
        if np.random.random <= Pxy(R1(q_prime,q)):
            return True
    return False

# ================================ TYPE 2 ---> 1 ================================

def R21(A12,A21,q0,q1,q,chi):
    acceptance_ratio = heaviside(q0,q1) * ( (A12) / (A21) ) * ( (np.pi * abs(u(q))) / (2*q0*abs(u(np.sqrt((q**2) + (q1**2) - 2*q*q1*chi)) * f(q1))) )
    return acceptance_ratio

def Type1to2(A12,A21,q0,q1,q,chi):
    '''
    q is taken from type 2 diagram
    q1 and chi are seeded from (0,q0) and (-1,1) respectively.
    '''
    if np.random.random() <= A21:
        if np.random.random() <= Pxy(R21(A12,A21,q0,q1,q,chi)):
            return True
    return False

# ================================ TYPE 2 ---> VARIABLE ================================

def R2(q,q1,qp,qp1,chi):
    '''
    qp = q prime
    qp1 = q prime 1
    '''
    acceptance_ratio = (abs(u(np.sqrt(qp**2 + qp1**2 - 2*qp*qp1*chi)) * f(qp1)) * q*q1) / (abs(u(np.sqrt(q**2 + q1**2 - 2*q*q1*chi)) * f(q1)) * qp*qp1)
    return acceptance_ratio

def variables2(A21,lam,q,q1,chi):
    if np.random.random() <= (1 - A21):
        qp = lambda_seed(lam,q)
        qp1 = lambda_seed(lam,q1)
        if np.random.random <= Pxy(R2(q,q1,qp,qp1,chi)):
            return True
    return False

# ================================ UPDATE FOR TYPE 0 ================================



# ================================ UPDATE FOR TYPE 1 ================================



# ================================ UPDATE FOR TYPE 2 ================================





# ================================ ALGORITHM ================================