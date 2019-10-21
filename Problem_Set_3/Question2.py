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

def get_spectrum_fix(tau, pars, lmax=1500):
    """Give the power spectrum of the CMB using the camb package. 
    -Arguments: -tau  (array): The value of tau (not with pars so that it can be kept fixed). MUST BE AN ARRAY!!!
                -pars (array): The parmaters for the model
                -lmax (array): Number of point at which we evaluate the function
    -Returns:   -tt   (array): The power spectrum evaluated up to around lmax"""
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau = tau[0]
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
    grad = np.zeros([len(data),len(p)])
    p_update = p.copy()
    for i in range(len(p)):
        dx = p_update[i]*0.001
        p_update[i] = p_update[i]+dx
        func_pdx = fit(p_update)[2:len(data)+2]
        p_update[i] = p_update[i]-2*dx
        func_mdx = fit(p_update)[2:len(data)+2]
        deriv = (func_pdx-func_mdx)
        deriv /= 2*dx
        grad[:,i] = np.array(deriv) 
    return grad


def newton_method_Levenberg(data,fit,p, error):
    """Performs Levenberg-Marquardt's method to find the best fit. Perform up-to 100 iterations. If chi_squared - chi_squared_new <1,
    stops the loop. If chi_squared>chi_squared new, increase lambda to use grhadient descent for a few stpes. If chi_square gets
    better,  
    -Arguments: -data (array): Data point to fit
                -fit  (array): The function that we want to optimize
                -p    (array): The paramters we want to optimize for the CMB power spectrum 
                -error(array): The error on the data points
    -Returns:   -p    (array): The  updated paramters for CMB power spectrum
                -pcov (array): Covariance matrix of the fit
                -perr (array): The error on the parameters
    NOTE: P MUST BE AN ARRAY EVEN IF YOU ONLY EVOLVE ONE PARAMETER!"""
    chi_squared = 10000
    lam = 0.0001
    grad = gradient(data,fit,p)
    for j in range(100):
        guess = fit(p)[2:len(data)+2]
        r=data-guess
        chi_squared_new=(r**2/error**2).sum()
        if (chi_squared - chi_squared_new) < 0.001 and (chi_squared - chi_squared_new) > 0:
            print(f'The final chi-squared is {chi_squared_new}\n')
            noise = np.diag(error**2)
            pcov = np.dot(grad.transpose(),noise)
            pcov = np.dot(pcov, grad)
            pcov = np.linalg.inv(pcov)
            perr = np.sqrt(np.diag(pcov))
            break
        elif chi_squared < chi_squared_new:
            print(f"Chi-Squared ({chi_squared_new})  got bigger, increasing lambda!!!")
            lam = 5*lam
        else:
            chi_squared = chi_squared_new
            lam /= 3
            grad = gradient(data,fit,p)
            r=np.matrix(r).transpose()
            grad=np.matrix(grad)
        lhs=grad.transpose()*grad
        lhs+=lam*np.diagonal(lhs)
        rhs=grad.transpose()*r
        dp=np.linalg.inv(lhs)*(rhs)
        for jj in range(p.size):
            p[jj]=p[jj]+dp[jj]
        print(f'Iteration {j}: chi-squared = {chi_squared}')
    return p,pcov,perr 

pars= np.asarray([65,0.02,0.1,2e-9,0.96])
tau = np.array([0.05])
f =lambda pars: get_spectrum_fix(tau,pars)
print(f'OPTIMIZATION WITH TAU FIX:')
pars,pcov,perr = newton_method_Levenberg(wmap[:,1], f, pars, wmap[:,2])
print(f'OPTIMIZATION FOR TAU FLOATING:')
f =lambda tau: get_spectrum_fix(tau, pars)
tau,tau_cov,tau_err = (newton_method_Levenberg(wmap[:,1], f, tau, wmap[:,2]))

pars = np.insert(pars,3,tau)
plt.plot(wmap[:,0],wmap[:,1],'.')
plt.plot(wmap[:,0], get_spectrum(pars)[2:len(wmap[:,0])+2])
plt.show()