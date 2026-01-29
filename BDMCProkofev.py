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
'''
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def Iu(qvals,deltaq):
    '''
    This is the integral from 0 to infinity of the absolute value of u(q) dq,
    due to the absolute value, this has no nice analytical expression. Therefore,
    an approximation with sum and infinity going to the max q value multiplied by
    deltaq
    '''
    I_sum = 0
    for q in qvals:
        I_sum += abs(u(q)) * deltaq
    return I_sum

@njit
def HistogramSum(H):
    '''
    Sums each bin in a histogram, NOTE: it returns the absolute value of each bin summed.
    '''
    return sum([abs(s) for s in H])

@njit
def u(q,potential):
    '''
    For a spherically symmetrical potential that is radius=1
    '''
    return ((2 * potential)/(q ** 3)) * (np.sin(q) - q * np.cos(q))

@njit
def RAA():
    return 1

@njit
def RAB(lam,):
    numerator1 = 2 * abs(1 + lam)
    numerator2 = 
    denominator1 = 
    denominator2 =  
