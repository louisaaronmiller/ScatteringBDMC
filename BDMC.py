import numpy as np

'''
x,y are integers that represent the type you are in.

Axy: Addressal probability to transfer from type x to type y.
Pxy: Acceptance probability of transfering from type x to type y.
Rxy: Acceptance ratio, that is used in the acceptance probability.
Oxy: Function that you seed with, O is for Omega.

U: update -> works for all types (type is in function signature, thus the function is called with a type)

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

def u(q,potential,r_star,mass,eps = 1e-5):
    if q <= eps:
        return u_0(potential,r_star,mass) * (1 - (q**2)/(2))
    return (3 * u_0(potential,r_star,mass)) * (np.sin(q) - q * np.cos(q))/(q ** 3)

def Pxy(Rxy):
    '''
    Generalised function for the acceptance probability using acceptance ratios
    x type -> y type
    '''
    return min(1,Rxy)

def get_bin(q,deltaq):
    return int(q/deltaq)



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

def R12(A12,A21,q0,q1,q,chi,deltaq,H):
    acceptance_ratio = ( (A21) / (A12) ) * ( (2*q0*abs(u(np.sqrt((q**2) + (q1**2) - 2*q*q1*chi)) * H[get_bin(q1,deltaq)])) / (np.pi * abs(u(q))) )
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
            return True,q_prime
    return False

# ================================ TYPE 2 ---> 1 ================================

def R21(A12,A21,q0,q1,q,chi,deltaq,H):
    acceptance_ratio = heaviside(q0,q1) * ( (A12) / (A21) ) * ( (np.pi * abs(u(q))) / (2*q0*abs(u(np.sqrt((q**2) + (q1**2) - 2*q*q1*chi)) * H[get_bin(q1,deltaq)])) )
    return acceptance_ratio

def Type2to1(A12,A21,q0,q1,q,chi):
    '''
    q is taken from type 2 diagram
    q1 and chi are seeded from (0,q0) and (-1,1) respectively.
    '''
    if np.random.random() <= A21:
        if np.random.random() <= Pxy(R21(A12,A21,q0,q1,q,chi)):
            return True
    return False

# ================================ TYPE 2 ---> VARIABLE ================================

def R2(q,q1,qp,qp1,chi,deltaq,H):
    '''
    qp = q prime
    qp1 = q prime 1
    '''
    acceptance_ratio = (abs(u(np.sqrt(qp**2 + qp1**2 - 2*qp*qp1*chi)) * H[get_bin(qp1,deltaq)]) * q*q1) / (abs(u(np.sqrt(q**2 + q1**2 - 2*q*q1*chi)) * H[get_bin(q1,deltaq)]) * qp*qp1)
    return acceptance_ratio

def variables2(A21,lam,q,q1,chi):
    if np.random.random() <= (1 - A21):
        qp = lambda_seed(lam,q)
        qp1 = lambda_seed(lam,q1)
        if np.random.random <= Pxy(R2(q,q1,qp,qp1,chi)):
            return True,qp,qp1
    return False

# ================================ UPDATE FOR TYPE 0 ================================
'''
Nothing happens, the update in type zero is skipped, the only thing you can do is to try to change to type 1
with a addressal probability of 1 and a acceptance probability of Pxy(R01).
'''
# ================================ UPDATE FOR TYPE 1 AND 2 ================================
def U(type:int,q,H,deltaq):
    '''    
    :Type: 0,1,2
    :q: momentum bin
    :H: Histogram
    '''
    bin_q = get_bin(q,deltaq)
    if type == 1:
        H[bin_q] += 1
    elif type == 2:
        H[bin_q] -= 1
    return H

# ================================ ALGORITHM ================================

def DiagMC(N,A10,A12,A21,q0,D0,deltaq,lam):

    q,q1,chi = seed(q0)

    Nbins = int(q0 / deltaq)
    H = np.ones(Nbins) * 1e-6   
    Type = 1 # Starting in type 1

    type0sum = 0
    type1sum = 1
    type2sum = 0

    for _ in range(N):



        if Type == 0:
            if Type0to1(q,q0,D0,A01=1):
                Type = 1
                type0sum += 1
                continue
            else:
                type0sum += 1
                continue



        if Type == 1:
            if Type1to0(A10,q0,q,D0):
                Type = 0
                type1sum += 1
                continue

            elif Type1to2(A12,A21,q0,q1,q,chi):
                Type = 2
                type1sum += 1
                continue

            else:
                _,q = variables1(A12,A21,q0,q1,q,chi)
                H = U(1,q,H,deltaq)
                type1sum += 1
                continue



        if Type == 2:
            if Type2to1(A12,A21,q0,q1,q,chi):
                Type = 1
                type2sum += 1
                continue

            else:
                _,q,q1 = variables2(A21,lam,q,q1,chi)
                H = U(2,q,H,deltaq)
                type2sum += 1
                continue
    total = type0sum + type1sum + type2sum
    return H,type0sum,type1sum,type2sum,total

    




# fix all calls to u(q) to suit needs to new u() that includes r_star
# whatever i need to do to fix r_star, and find out what to do about it
# fix R12,R21, R2 such that they all recieve H,delta in there function signatures.
# in type1, add a random number instead of calling it, since there are two moves that take place, a single random number is better
# same for type2
# just like i update q,q1, i need to update chi from type2


