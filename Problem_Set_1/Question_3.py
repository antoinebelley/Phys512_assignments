#Test the adaptative simpson_rule for different funcitons
#REMARK: Requires python 3.6 or higher due to the use of f-strings
#Author: Antoine Belley
#15/09/19

import numpy as np
from adaptative_simpson_rule import integrate_adaptative
from scipy.special import sici


def func():
    """Test function 1/1+x
    Returns: -f    (function) The fucntion to be integrated
             -name (string)   Function form in string
             -inte (fucntion) Analytic solution of the integral"""
    f = lambda x: 1/(1+x)
    #f = lambda x: catch_ZeroDivision_inLambda(f,x)
    inte = lambda x: np.log(1+x)
    return f,  '1/(1+x)', inte


def func2():
    """Test fucntion sin(x)/x
    Returns: -f    (function) The fucntion to be integrated
             -name (string)   Function form in string
             -inte (fucntion) Analytic solution of the integral"""
    f = lambda x: np.sin(x)/x
    inte = lambda x: sici(x)[0]
    return f , "sin(x)/x", inte


def func3():
    """Lorentzian function
    Returns: -f    (function) The fucntion to be integrated
             -name (string)   Function form in string
             -inte (fucntion) Analytic solution of the integral"""
    f = lambda x: 1/ np.pi / (x**2)
    inte = lambda x: -1/np.pi/x
    return f, "a lorentzian", inte

funcs = [func(), func2(), func3()]


def test_integrals(funcs, a, b, tol=1e-7, nmax=1000):
    """Tests different integrals
    Arguments: - funcs (array of function) The fucntion to be integrated
               - a     (float)             Lower bound of the integral
               - b     (float)             Upper bound of the integral
               - tol   (float)             Tolerance for the integral
               - nmax  (int)               Max number steps for the integral to converge"""
    for f in funcs:
        integ, count = integrate_adaptative(f[0] ,a ,b , tol , nmax)
        error = np.abs(integ - f[2](b)+f[2](a))
        print(f'The integral of {f[1]} from 1 to 5 converged to {integ} in {count} steps with an error of {error}.')
        print(f'The real value is of {f[2](b)-f[2](a)}')
        print(f'We would need {5*count} funciton evaluation with the code we did in class but the "smart" way allows for only {2*count} function evaluation\n')
    print()


test_integrals(funcs, 0.1,1)