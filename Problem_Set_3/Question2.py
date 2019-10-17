#Question 2 of problem set 3 in phys 512
#We want to optimize the parameters for the model
#using a Levenberg- Marquardt minimizer. 
#Author : Antoine Belley
#Date:    17/10/2019
#WARNING: needs python 3.6 or higher due to the use of f-strings


import numpy as np
from scipy.interpolate import interp1d
from wmap_camb_example import get_spectrum, wmap, pars


def derivative(function, x, dx):
    """Calculate the derivative of a fucntion using the higher-order 
    shceme detailed in Question_1.pdf
    Arguments: -fucntion : (function)  The function to be derivated
               -x        : (float)     Point at wich the function is being derivated
               -dx       : (float)     Step-size for the derivative
    Returns:   -fp       : (float)     Derivatives of the point evaluated at x"""
    fp = 8.0*(function(x+dx)-function(x-dx)) - (function(x+2.0*dx)-function(x-2.0*dx))
    fp /= 12.0*dx
    return fp



def newton_method_Levenberg(data,fit,p, fix = None):
    """Performs Newton's method to find the best fit. Perform 5 iterations. If chi_squared < 1 and 
    reduces by less than 0.1% of its value, stops the iteration.
    -Arguments: -data            (array): Data point to fit
                -x               (array): The x-value at which the function is evaluated
                -p               (array): The paramters for the exponential in the order [x0,a,b,c]
    -Returns:   -p               (array): The  updated paramters for the exponential in the order [x0,a,b,c]
                -chi_squared_new (array): The chi-square of the best fit"""
    chi_squared = 1000
    for j in range(5):
        guess = fit(p)
        x = np.arange(0,len(guess),1)
        model = interp1d(x,guess, kind = 'cubic')
        x = np.array(wmap[:,0])
        guess  = model(x)
        r=data-guess
        chi_squared_new=(r**2).sum()
        r=np.matrix(r).transpose()
        grad=np.matrix(grad)
        lhs=grad.transpose()*grad
        rhs=grad.transpose()*r
        dp=np.linalg.inv(lhs)*(rhs)
        for jj in range(p.size):
            p[jj]=p[jj]+dp[jj]
        #Stop iterating if the chi-square is small enough 
        if chi_squared_new < 1 and  chi_squared_new - chi_squared/chi_squared < 1 and j>0:
            break
        else:
            chi_squared = chi_squared_new

newton_method_Levenberg(wmap[:,1],get_spectrum,pars)