#Plot the error (due to floating point precision) of the derivative of a function 
#as I change the step-size. The derivative is a scheme that recuded 
#the truncation error to the 5th order. (See pdf Question_1.pdf for detail on the shceme)
#Requires python 3.6 or higher (because of the use of f-strings)
#Author: Antoine Belley
#14/09/19


import numpy as np
import matplotlib.pyplot as plt

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


def plot_error_dx(function, x, a):
    """Plot and save the figure showing the error due to floating points as a function of dx
    Arguments: -fucntion   : (function) The function to be derivated
               -x          : (flaot)    Point at wich the function is being derivated
               -a          : (float)    Factor in the exponent"""

    def true_deriv(x, a):
        """Computes the analytic derivatives for 
        functions of the form e^(ax)
        Arguments: -x          : (flaot)    Point at wich the function is being derivated
                   -a          : (float)    Factor in the exponent 
        Returns:   -deriv      : (float)    Analytic derivative evaluated at x"""
        return a*np.exp(a*x)

    def name(a):
        """Gives a string name to the fucntion for the title and the savefig
        Arguments: -a          : (float)  Factor in the exponent
        Returns:   -name       : (string) Function in string form"""
        return f'e^({a}x)'

    dx = np.logspace(-14,1 , 100)
    points = []
    for i in dx:
        error = true_deriv(x,a) - derivative(function, x, i)
        points.append(error)
    points = np.array(points)
    points = np.abs(points)
    plt.figure()
    plt.plot(dx, points, marker='*')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('dx')
    plt.ylabel('Error')
    plt.title(f'Error due to floating point for {name(a)}')
    plt.savefig(f'error_{name(a)}.pdf')


e = np.exp
def e2(x):
    return np.exp(0.01*x)

plot_error_dx(e, 0, 1)
plot_error_dx(e2, 0, 0.01)





