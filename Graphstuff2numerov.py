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
    V_0 = np.linspace(-33, -0.1, 2000)
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
        if idx % 100 == 0:
            print(idx)

    scattering_length = -delta / k_arr
    V_0abs = np.abs(V_0)

    return delta, V_0, k_arr, sigmas, scattering_length, V_0abs

@njit
def AnSL(a):
    V_0 = np.linspace(33,0.1,10000) ##### MATCH----------------------
    SLarr = []
    for i in V_0:
        gamma = (np.sqrt(2)*i**0.5) * a
        SL = (1 - (np.tan(gamma))/(gamma)) * a
        SLarr.append(SL)
    return V_0,SLarr

def wavefunctionunshifted(l: int, E: float, rmax, R, V_0, h = 0.01):
    u, r =  Numerov(l=l, E=E,rmax = rmax,R=R, h = 0.01,V_0=V_0)
    return u,r


delta, V_0, k, sigmas, scattering_length,V_0abs = ScatterLength(l=0,E=1e-4,rmax=600
                                                         ,R=1)

V_0an, SLan = AnSL(1)

plt.plot(V_0abs ,scattering_length,color = 'red',label = 'Numerov')
plt.plot(V_0an,SLan,'--k',label = 'Analytical')
plt.xlabel('$|V|$')
plt.ylabel('Scattering Length')


plt.text(3, 20, '$\\gamma = \\frac{\\pi}{2}$', fontsize=17)
plt.text(13, 20, "$\\gamma = \\frac{3\\pi}{2}$", fontsize=17)
plt.text(25, 20, "$\\gamma = \\frac{5\\pi}{2}$", fontsize=17)
plt.legend(loc='lower right')
plt.minorticks_on()
plt.ylim(-30,40)

plt.grid()
plt.show()


















# -------------------------------------------------------------------------------------------------------------------------





















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
def F(l, r, E,R=R,V_0=V_0):
    if r == 0:
        return 0
    else:
        func_val = V(r,R,V_0) + (l * (l + 1)) / (r**2) - E
        return func_val

@njit
def k(E, r,R=R,hbar = 1):
    if r > R:
        return math.sqrt(2 *E)/hbar
    else:
        return math.sqrt(E - V_0)
    
@njit
def g_func(l, r, E, R, V_0, m, hbar):
    # avoid r=0 singularity (you already start with u[0]=0 anyway)
    if r == 0.0:
        return 0.0

    # g(r) = (2m/hbar^2)*(V - E) + l(l+1)/r^2
    pref = (2.0 * m) / (hbar * hbar)
    return pref * (V(r, R, V_0) - E) + (l * (l + 1.0)) / (r * r)


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

def outside_vals(rvals, uvals,R=R):
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


def r_1halfr_2(r, u, E, max_points=2):
    """
    Finds r and u values at successive extrema (half-wavelength separation).
    Works by finding local maxima/minima in u(r).
    """

    u = np.array(u)
    r = np.array(r)

    # Finding indices of local maxima and minima
    extrema_idx = (
        argrelextrema(u, np.greater)[0].tolist() + argrelextrema(u, np.less)[0].tolist()
    )
    extrema_idx.sort()

    extrema_idx = extrema_idx[:max_points]

    r_aug = r[extrema_idx]
    u_aug = u[extrema_idx]

    return r_aug, u_aug


def wavefunctionunshifted(l: int, E: float, rmax, R, V_0, h = 0.01):
    u, r =  Numerov(l=l, E=E,rmax = rmax,R=R, h = h,V_0=V_0)
    return u,r


eee = 0.01
u1,rr1 = wavefunctionunshifted(l=0, E=eee, rmax = 200,R= 1,V_0= 0,h = 0.001)
u2,rr2 = wavefunctionunshifted(l=0, E=eee, rmax = 200,R= 1,V_0= -15,h = 0.001)
u3,rr3 = wavefunctionunshifted(l=0, E=eee, rmax = 200,R= 1,V_0= -65,h = 0.001)
u4,rr4 = wavefunctionunshifted(l=0, E=eee, rmax = 600,R= 1,V_0= -60.43,h = 0.001)


import numpy as np
import matplotlib.pyplot as plt

X = 4
Y = 54
a = 1.0
V0 = 30

r2 = np.linspace(0,1,200)
r = np.linspace(0,X,1000)

psi = np.sin(np.pi*r2/a)
psi2 = np.sin(2*np.pi*r2/a)
psi3 = np.sin(3*np.pi*r2/a)
psi4 = np.sin(4*np.pi*r2/a)

V = np.zeros_like(r2)
V[r2 < a] = -V0
V[0] = 0
V[-1] = -0.15

fig, ax = plt.subplots()

# Plot well
ax.plot(r2, V, color='black', linewidth=0.90)

# Bound energies
bound1, bound2, bound3,bound4 = -25.71, -17.14, -8.57 ,0
bound_energies = [bound1, bound2, bound3]

# Scattering axis levels
axis1, axis2, axis3 = 9, 27, 45
axii = [axis1, axis2, axis3]

# Plot wavefunctions in well
ax.plot(r2, psi*3 + bound1, color='k')
ax.plot(r2, psi2*3 + bound2, color='k')
ax.plot(r2, psi3*3 + bound3, color='k')
ax.plot(r2, psi4*3 + bound4, color='k')
# Plotting numerov wave


#ax.plot(rr1,(u1)*12 + axis1,color='k')
#ax.plot(rr2,(u2)*16 + axis2,color='k')
ax.plot(rr3,(u3)*30 + axis1 * 4,color='k')
ax.plot(rr4,(u4)*30 + axis2- 9,color='k')

# Bound energy dashed lines
for E in bound_energies:
    ax.hlines(E, 0, a, colors='k', linestyles='--', linewidth=0.9)
'''
# Positive energy dashed lines
for E in axii:
    ax.hlines(E, 0, X, colors='k', linestyles='--', linewidth=0.9)
'''
ax.hlines(axis2 -9,0,X,colors='k',linestyles='--', linewidth=0.9)
ax.hlines(axis1*4,0,X,colors='k',linestyles='--', linewidth=0.9)

ax.set_xlabel("$r$")
ax.set_xlim(0, X+50)

current_lower = ax.get_ylim()[0]
ax.set_ylim(current_lower, Y)

ax.spines['left'].set_bounds(-V0, Y)

# Move x-axis to V=0
ax.spines['bottom'].set_position(('data',0))

# Clean frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Remove tick marks
ax.set_xticks([])
ax.set_yticks([])

# Label for potential
ax.text(1.1, bound2, "$-V$")
ax.fill_between([0, a], -V0, Y, color='grey', alpha=0.3)
fig.set_size_inches(5, 7)
'''
ax.text(3.5, axis1 -5 , "$V = 0$")
ax.text(3.5, axis2 -10 , "$V = 15$")
ax.text(3.5, axis3 - 10, "$V = 160$")
'''

ax.text(3, axis2 -5.5 , "$V \\approx 60.43$")
ax.text(2.705, axis2 -7.5 , "Fourth bound state")


fig.subplots_adjust(
    left=0.05,
    right=0.95,
    bottom=0,
    top=0.95,
    wspace=0.3,
    hspace=0.3
)



plt.show()
