#Question 3 of problem set 3 in phys 512
#We want to optimize the parameters for the model
#using a MCMC 
#Author : Antoine Belley
#Date:    17/10/2019
#WARNING: needs python 3.6 or higher due to the use of f-strings

import numpy as np
import camb
from wmap_camb_example import wmap, get_spectrum
import matplotlib.pyplot as plt

#file = open('Final_result.txt', 'a')
#file.write('QUESTION 3\n')
#file.write('-----------\n')

#The covariance matrix copied from Question 2
pcov = np.array([[ 1.34451562e+01,  2.57312294e-03, -2.39792642e-02,  4.14041345e-01,
   1.57967658e-09,  8.47758758e-02],
 [ 2.57312294e-03,  7.24532574e-07, -3.70603312e-06,  9.82776311e-05,
   3.91549724e-13,  2.08178230e-05],
 [-2.39792642e-02, -3.70603312e-06,  4.95035503e-05, -7.04182401e-04,
  -2.60506065e-12, -1.36513840e-04],
 [ 4.14041345e-01,  9.82776311e-05, -7.04182401e-04,  2.23170602e-02,
   8.84945934e-11,  3.34483347e-03],
 [ 1.57967658e-09,  3.91549724e-13, -2.60506065e-12,  8.84945934e-11,
   3.52431715e-19,  1.32130403e-11],
 [ 8.47758758e-02,  2.08178230e-05, -1.36513840e-04,  3.34483347e-03,
   1.32130403e-11,  6.85759934e-04]])

def take_step_cov(covmat):
    """Function to give the step size for the parameters for the MCMC
       Arguments: -covmat (array): covariance matrix of the parameter
       Returns:   -step   (array): Vector containing the step size for each parameter"""
    mychol=np.linalg.cholesky(covmat)
    return np.dot(mychol,np.random.randn(covmat.shape[0]))

x=wmap[:,0]
pars=np.asarray([67,0.02,0.1,0.1,2e-9,0.96])

nstep=5000
npar=len(pars)
noise = wmap[:,2]
chains=np.zeros([nstep,npar])
chisq=np.sum((wmap[:,1]-get_spectrum(pars)[2:len(x)+2])**2/noise**2)
scale_fac=0.5
chisqvec_new=np.zeros(nstep)
file = open('chains_Question3_2.txt','w')
count = 0
for i in range(nstep):
    new_params=pars+take_step_cov(pcov)*scale_fac
    if new_params[3]>0:
      new_model=get_spectrum(new_params)[2:len(x)+2]
      new_chisq=np.sum((wmap[:,1]-new_model)**2/noise**2)
      delta_chisq=new_chisq-chisq
      prob=np.exp(-0.5*delta_chisq)
      accept=np.random.rand(1)<prob
      if accept:
          print(f'Accepted {count}')
          count+=1
          pars=new_params
          chisq=new_chisq
    chains[i,:]=pars
    for j in pars:
        file.write(f'{j} ')
    file.write(f'{chisq}\n')
    file.flush()
    chisqvec_new[i]=chisq
file.close()  

fit_params=np.mean(chains,axis=0)

