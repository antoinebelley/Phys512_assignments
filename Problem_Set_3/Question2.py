#Question 2 of problem set 3 in phys 512
#We want to optimize the parameters for the model
#using a Levenberg- Marquardt minimizer. 
#Author : Antoine Belley
#Date:    17/10/2019
#WARNING: needs python 3.6 or higher due to the use of f-strings


import numpy as np
import camb
from wmap_camb_example import wmap, get_spectrum
import matplotlib.pyplot as plt

file = open('Final_result.txt', 'a')
print('QUESTION 2\n')
print('-----------\n')


def get_spectrum_fix(pars, lmax=1500):
    """Give the power spectrum of the CMB using the camb package with tau fixed.
    -Arguments: -pars (array): The parmaters for the model that will be updated
                -lmax (array): Number of point at which we evaluate the function
    -Returns:   -tt   (array): The power spectrum evaluated up to around lmax"""
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau = 0.001
    As=pars[3]
    ns=pars[4]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt


def gradient(data,fit,p):
    """Computes the gradient of the fit function with respect to the parameters p.
    -Arguments: -data (array): Data point (just given to get gradient of the right length)
                -fit  (array): The function that we want to compute the gradient of
                -p    (array): The paramters with respect to which we want to take the derivatives
    -Returns:   -grad (array): The gradient of the function"""
    grad = np.zeros([len(data),len(p)])
    for i in range(len(p)):
        p_update = p.copy()
        dx = p_update[i]*0.05
        p_update[i] = p_update[i]+dx
        func_pdx = fit(p_update)[2:len(data)+2]
        p_update[i] = p_update[i]-2*dx
        func_mdx = fit(p_update)[2:len(data)+2]
        deriv = (func_pdx-func_mdx)
        deriv /= 2*dx
        grad[:,i] = np.array(deriv) 
    return grad


def newton_method_Levenberg(data,fit,p, error, iter_max = 100):
    """Performs Levenberg-Marquardt's method to find the best fit. If chi_squared - chi_squared_new <1,
    stops the loop. If chi_squared>chi_squared new, increase lam to use grhadient descent for a few stpes. If chi_square gets
    better, reduces lam to proceed with Newton's method.  
    -Arguments: -data (array): Data point to fit
                -fit  (array): The function that we want to optimize
                -p    (array): The paramters we want to optimize for the CMB power spectrum 
                -error(array): The error on the data points
                -iter_max (int): Maximal number of steps for the optimization process, default = 100.
    -Returns:   -p    (array): The  updated paramters for CMB power spectrum
                -pcov (array): Covariance matrix of the fit
                -perr (array): The error on the parameters
    NOTE: P MUST BE AN ARRAY EVEN IF YOU ONLY EVOLVE ONE PARAMETER!"""
    chi_squared = 10000 #Random big value for comparison
    lam = 0.0001        #Smart value to begin with
    grad = gradient(data,fit,p)
    p_new = p.copy()
    for j in range(iter_max):
        guess = fit(p_new)[2:len(data)+2]
        r=data-guess
        chi_squared_new=(r**2/error**2).sum()
        if (chi_squared - chi_squared_new) < 0.001 and (chi_squared - chi_squared_new) > 0:
            #Stops loops if chi_squared - chi_squared_new <1
            p = p_new.copy()
            print(f'The final chi-squared is {chi_squared_new}\n\n')
            #Add back tau to include it in the covariance matrix
            p = np.insert(p,3,0.05)
            grad = gradient(data, get_spectrum,p)
            noise = np.diag(1/error**2)
            pcov = np.dot(grad.transpose(),noise)
            pcov = np.dot(pcov, grad)
            pcov = np.linalg.inv(pcov)
            perr = np.sqrt(np.diag(pcov))
            break
        elif chi_squared < chi_squared_new:
            #If step increases the chi_squared, increases lam to switch to gradient descent
            print(f"Chi-Squared ({chi_squared_new})  got bigger, increasing lambda!!!\n")
            lam *= 1000
            p_new=p.copy()
        else:
            #Performs Newton's method
            chi_squared = chi_squared_new
            p = p_new.copy()
            lam /= 700
            grad = gradient(data,fit,p)
        r=np.matrix(r).transpose()
        grad=np.matrix(grad)
        JJ=grad.transpose()*np.diag(1/error**2)*grad
        lhs=JJ+lam*np.diag(np.diag(JJ))
        rhs=grad.transpose()*np.diag(1/error**2)*r
        dp=np.linalg.inv(lhs)*(rhs)
        for jj in range(p.size):
            p_new[jj]=p_new[jj]+dp[jj]
        print(f'Iteration {j}: chi-squared = {chi_squared}\n')
    return p,pcov,perr 


pars= np.asarray([65,0.02,0.1,2e-9,0.96])   #Initial values of the parameters without tau
tau = np.array([0.001])                      #Initial value of tau      #Function with pars floating and tau fix
print(f'OPTIMIZATION WITH TAU FIX:\n')
pars,pcov,perr = newton_method_Levenberg(wmap[:,1], get_spectrum_fix, pars, wmap[:,2]) #Optimization with tau fix
#Write the final parameters and their errors in the result file 
print('The final parameters are:\n')
print(f'\t-H0    = {pars[0]} pm {perr[0]}\n')
print(f'\t-ombh2 = {pars[1]} pm {perr[1]}\n')
print(f'\t-omch2 = {pars[2]} pm {perr[2]}\n')
print(f'\t-tau   = {pars[3]}  pm {perr[3]}\n')
print(f'\t-As    = {pars[4]} pm {perr[4]}\n')
print(f'\t-ns    = {pars[5]} pm {perr[5]}\n\n')

print('We see that the Newton gives decreasing value of chi-squared so the derivatives\n')
print('can be trusted!\n\n')

print('We see from looking at the covariance matrix that tau is not really correlated to\n')
print('the other parameters since its error is very big after updating the new parameters\n')
print('so floating tau would not really affect the errors of the other parameters.\n\n')

print(pcov)




