#Procedes to do a Chebyshev fit of a function and compare it
#to a linear fit. 
#Author: Antoine Belley
#Date: 29/09/19


import numpy as np
import matplotlib.pyplot as plt

def cheby_fit(x, y, order, tolerance):
    """Makes a Chebyshev fit to a fucntion up to a certain order and certain 
    tolerance. I.e. it will fit until it reaches the tolerance or it reaches the maximum 
    order inputed. Once it generated the Chebyshev polynomials, it uses a least-square to
    find the coefficiens in front of the polynomials.
    -Arguments: - x       (array): x-value of the data to fit
                - y       (array): y-value of the data to fit
                - order     (int): maximal order for the fit
                - tolerance (int): tolerance on the error of the fit
    -Returns:   - fit     (array): The fitted points to the data
                - max_coeff (int): The number of terms needed to get the required tolerance"""


    def cheby_poly(x, order):
        """Genrates the Chebyshev polynomials up to the desire order
        -Arguments: - x    (array): The x-array of the data to fit (must be in [-1,1])
                    - order  (int): Maximal order of the polynomials we want to generate
        -Return:    - poly (array): Chebyshevs polynomials in an array"""

        poly = np.zeros([len(x),order+1]) #Generates the array into which we will input the polynomials
        poly[:,0] = 1 #Initiate Chebyshev polynomial of order 0
        if order > 0:
            poly[:,1] = x #Initiate Chebyshev polynomial of order 1
        if order > 1:
            for i in range(1,order):
                #Initiate the polynomials of higher order using the following recursion:
                #T_{n+1}(x) = 2xT_{n} - T_{n-1}
                poly[:,i+1] = 2*x*poly[:,i]-poly[:,i-1]
        return poly

    def map_x_range(x):
        """Maps the value of x into the interval (-1,1) using the transfomation
        x_new = -1 + ((x - a1) * (1 - -1) / (a2 - a1))
        - Arguments: - x     (array): The x-array of the data to map
        - Returns:   - x_new (array): Value of x mapped in (-1,1)"""
        x_new = np.zeros(len(x))
        for i in range(len(x)):
            x_new[i] =  -1 + ((x[i] - x[0]) * 2 / (x[-1] - x[0]))
        return x_new

    
    def least_square_coeff(poly, y):
        """Makes a least-square fit to find the optimal coefficient for the fit 
        of the Chebyshev polynomials.
        -Arguments: poly (array): The Chebyshev polynomials
                    y    (array): The data to be fit
        -Returns  : fit  (array): The expected values given by the fit"""
        lhs   = np.dot(poly.transpose(), poly)
        rhs   = np.dot(poly.transpose(), y)
        coeff = np.dot(np.linalg.inv(lhs), rhs)
        return coeff


    def coeff_tolerance(coeff, tol):
        """Picks the coeffs so that the error is in the tolerance
        -Arguments: coeff (array): The Chebyshev polynomials
                    tol   (array): The data to be fit
        -Returns  : max_coeff  (array): The expected values given by the fit"""
        max_coeff = len(coeff)
        for i in range(len(coeff)):
            if np.absolute(coeff[i]) <= tol:
                max_coeff = i
                break
        try:
            return max_coeff
        except:
            print("Tolerance cannot be obatained at this order.")
            return len(coeff)

    x_new = map_x_range(x)
    polynomials = cheby_poly(x_new,order)
    coeff = least_square_coeff(polynomials,y)
    max_coeff = coeff_tolerance(coeff, tolerance)
    fit = np.dot(polynomials[:,:max_coeff], coeff[:max_coeff])
    return fit, max_coeff
    
#Change this so that it works for any positive interval
x = np.arange(0.5,1,1e-3)
y = np.log2(x)
fit, max_coeff = cheby_fit(x,y,50,1e-6)
fit_lin = np.polyfit(x,y,max_coeff-1)
fit_lin = np.polyval(fit_lin,x)

plt.plot(x,y,label='Function')
plt.plot(x,fit, label='cheb_fit')
plt.legend()
plt.savefig('Chebyshev_fit')
plt.clf()
plt.plot(x,fit-y, label = 'Chebyshev')
plt.plot(x,fit_lin-y, label = 'Polynomial')
plt.legend()
plt.title('Residual of the fits')
plt.savefig('Residuals')

print(f'The maximum error for the Chebyshev fit is of {np.max(fit-y)} and the RMS is of {np.sqrt(np.mean((fit-y)**2))}')
print(f'The maximum error for the polynomial fit is of {np.max(fit_lin-y)} and the RMS is of {np.sqrt(np.mean((fit_lin-y)**2))}')
print('NOTE: I did my mapping for a in a way that answered part b) at the same time!')

