#Compute and plot the electric field of a spehrical shell as a function of r
#by considering a ring and integrating over the rings in the shell.
#The integral can be found in the solution of Griffith 2.7
#It is given by 1/(2*e0)*(R^2*sig) ∫du (z-Ru)/(R^2+z^2-2Rzu)^(3/2)
#We compare the results given by the adaptative simpson's rule and 
#scipy.integrate.quad.
#WARNING: Rquires python 3.6 or higher due to the use of f-strings
#Author: Antoine Belley
#16/09/19


from scipy import integrate
from adaptative_simpson_rule import integrate_adaptative
import numpy as np
import matplotlib.pyplot as plt


def integrand(u,R,z):
    integrand = z-R*u
    integrand /= (R*R+z*z-2*R*z*u)**(3/2)
    return integrand


#We need to fix a value of R and to iterate over different values of z to 
#be able to get an idea of the shape of the E-field. Let choose R = 1 and z 
#from 0.001 to 2 going by steps of 0.001
def E_field_adaptative(a, b, tol, nmax):
    """Compute the E-field using of the function using the adaptative simpson rule
    Arguments: - a       (float) Lower bound of the integral
               - b       (float) Upper bound of the integral
               - tol     (float) Tolerance of the integration error
               - nmax    (int)   Maximal number of steps for the integral to converge
    Returns:   - E_field (array) Electric-field from 0 to 2"""
    R = 1
    z = np.linspace(0,2,2001)
    E_field = []
    for i in z:
        f = lambda x : integrand(x,R,i)
        #Error handling for errors coming from the division by 0
        try:
            point, counts = integrate_adaptative(f,a,b,tol,nmax)
        except:
            print(f"Integral couldn't be performed for u={i} due to division by 0 in the function and/or derivative.")
            print(f"Value of the intgeral at u = {i} is set for 0 for plotting purposes.")
            point = 0
        E_field.append(point)
    return E_field



def E_field_quad(a, b):
    """Compute the E-field using of the function using the quadrature
    Arguments: - a       (float) Lower bound of the integral
               - b       (float) Upper bound of the integral
    Returns:   - E_field (array) Electric-field from 0 to 2"""
    R = 1
    z = np.linspace(0,2,2001)
    E_field = []
    for i in z:
        f = lambda x : integrand(x,R,i)
        point, err = integrate.quad(f,a,b)
        E_field.append(point)
    return E_field


#Plot the field for quad and adaptative simpson's rule
plt.clf()
plt.plot(np.linspace(0,2,2001),E_field_quad(-1,1), label='Quad')
plt.plot(np.linspace(0,2,2001),E_field_adaptative(-1,1,1e-7,1000), label = 'Adaptative simpson rule')
plt.xlabel('z')
plt.ylabel(r'E-field in units of $\frac{\sigma}{2*\varepsilon_0}$')
plt.title('E-field of a sphere with R=1')
plt.legend()
plt.show()

