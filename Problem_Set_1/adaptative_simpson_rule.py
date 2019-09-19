#Integrate using the adaptative simpson rule in a recursive manner
#REMARK: Requires python 3.6 or higher due to the use of f-strings
#Author: Antoine Belley
#15/09/19

import numpy as np


def simpson_rule_half(f, a, fa, b, fb):
    """Integrate f(x) from a to be by splitting the iterval in n subinterval
    and using Simpson's rule 
    -Arguments:  - f:        (function)  Function to be integrated
                 - a:        (float)     Lower bound of the integral
                 - fa:       (float)     Value of f at a
                 - b:        (float)     Upper bound of the integral
                 - fb:       (float)     Value of f at b
    -Returns:    - integral: (float)     Value of the integral
                 - h:        (float)     Midpoint of the function
                 - fh:       (float)     Function evaluated at the mid point """
    if a == b:
        return 0
    if a > b:
        sign = -1
    else:
        sign = 1
    dx = (b-a)/(6) #(b-a)/(3*n) but here n is 2 since we are doing the half-rule
    h = (a+b)/2
    fh = f(h)
    integral = dx*sign*(fa+4*fh+fb)
    return integral, h, fh

def adaptative_simpson_rule(f, a, fa, b, fb, guess, h, fh, tol, nmax, count=0):
    """Estimatw of the error of the integral using 
    |S(a,h)+S(h,b)-S(a,b)| < 15*epsilon
    Where [a,b] is our interval, S() is the value of the simspson rule
    and epsilon is our tolerance. If the error is smaller than the tolerance
    it gives back the value of the interval, if not it splits the interval in halfs.
    -Arguments: - f:        (function)  Function to be integrated
                - a:        (float)     Lower bound of the integral
                - fa:       (float)     Value of f at a
                - b:        (float)     Upper bound of the integral
                - fb:       (float)     Value of f at b
                - guess:    (float)     Value of the integral given by Simpson's rule on whole interval
                - h:        (float)     Half-point of the interval
                - fh:       (float)     Value of the function evaluated at the half-point
                - tol:      (float)     Tolerrance on the precison of the result
    -Returns:   - integral: (Tuple)     Alternance of the value of the integral on the count"""
    if count < nmax:
        Sah, left_h, left_fh = simpson_rule_half(f, a, fa, h, fh)
        Shb, right_h, right_fh = simpson_rule_half(f, h, fh, b, fb)
        error = np.abs(Sah+Shb - guess)
        if error < 15*tol:
            return guess, count
        else:
            return adaptative_simpson_rule(f, a, fa, h, fh, Sah, left_h, left_fh, tol/2 ,nmax, count+1) + adaptative_simpson_rule(f, h, fh, b, fb, Shb, right_h, right_fh, tol/2,nmax, count+1)
    else:
        print(f'Integral did not converge in {nmax} steps. Increase namx or find a more suitable integration technique.')
        print('Exiting...')
        exit(1)


def integrate_adaptative(f, a, b, tol, nmax):
    """Integrate f(x) using the adaptative simpson rule. Sum the value on each subinterval
     and the total number of steps required to get the algortithm to converge
    -Arguments:  - f:        (function)  Function to be integrated
                 - a:        (float)     Lower bound of the integral
                 - b:        (float)     Upper bound of the integral
                 - tol:      (float)     Tolerance on the error of the integral
                 - nmax:     (int)       Maximum number of steps for the recursion
    -Returns:    - integral: (float)     Value of the integral
                 - count   : (int)       Number of steps for the fucntion to converge"""
    fa = f(a)
    fb = f(b)
    guess, h, fh = simpson_rule_half(f, a, fa, b, fb) 
    integral = np.array(adaptative_simpson_rule(f, a, fa, b, fb, guess, h, fh, tol, nmax))
    count    = np.sum(integral[1::2])
    integral = np.sum(integral[0::2])
    return integral, count