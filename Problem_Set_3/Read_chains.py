#Question 3 of problem set 3 in phys 512
#Reads of the results chains and plots the chains and their FFT to show their convergence
#Author : Antoine Belley
#Date:    17/10/2019
#WARNING: needs python 3.6 or higher due to the use of f-strings


import numpy as np
import matplotlib.pyplot as plt


def plot_chains(Question,file, burnin=500, FFT=False):
    """Plots the Markov chains
    -Arguments: -Question     (int): The number of the question to put in the names of the output files
                -file      (string): Name of the file where the chain are saved
                -burnin       (int): Number of points to remove before the chain converges. Default = 500.
                -FFT         (bool): If true, plots the fft of the chains
    -Returns:   -fit_params (array): The optimal parameters from the MCMC (Given by the mean of the chains)
                -errors     (array): Error on the fir parameters. (Given by the std of the chains)
    """

    chains = np.loadtxt(file)
    params = chains[burnin:,]


    names = ['H0', 'ombh2', 'omch2', 'tau', 'As', 'ns', 'chisq']
    fit_params=np.mean(params,axis=0)
    errors = np.std(params,axis=0)

    fig = plt.figure(figsize=(12,12), frameon=False)
    fig.patch.set_visible(False)
    plt.title(f'Chains of the question {Question}')
    for i in range(len(names)):
        ax = fig.add_subplot(len(names),1,i+1)
        ax.plot(chains[:,i])
        ax.axvspan(0,burnin,facecolor='r', alpha=0.5)
        ax.set_ylabel(names[i])
        ax.set_xlim(0,5000)
        ax.axhline(fit_params[i], label=f'Average value = {fit_params[i]}', color='orange')
        ax.legend(loc=1)
    plt.xlabel('Steps')
    plt.savefig(f'Question{Question}_chains.pdf')

    if FFT==True:
        fig = plt.figure(figsize=(12,12), frameon=False)
        plt.title(f'FFT of the chains of the question {Question}')
        for i in range(len(names)-1):
            ax = fig.add_subplot(len(names)-1,1,i+1)
            plt.plot(np.fft.rfft(chains[:,i]))
            plt.axvspan(0,burnin,facecolor='r', alpha=0.5)
            ax.set_ylabel(names[i])
            ax.set_yscale('symlog')
            ax.set_xscale('log')
        plt.xlabel('Steps')
        plt.savefig(f'Question{Question}_chains_FFT.pdf')        

    return fit_params,errors

def gaussian(x,mu,sig, sample):
    """Normalized gaussian for the importance sampling of the chain of question 3
       Arguments: -x     (array): The point at which we evaluate the gaussian
                  -mu     (real): Mean of the distribution
                  -sig    (real): Standard distribution of the distribution
                  -sample  (int): Number of sample in the distribution
       Returns:   -gauss (array): Returns the values of the gaussian evaluated at the points"""
    return 1/(sig*np.sqrt(2*np.pi*sample*0.01))*np.exp(-0.5*((x-mu)/sig)**2)

def prior_sample(file, burnin=500):
    """Take an importance sampling of the chains from question 3 wrt the experimental value of tau.
       Arguments: -file      (string): Name of the file of the chain that you want to importance sample
                  -burnin       (int): Number of points to remove before the chain converges. Default = 500.
       Returns:   -fit_params (array): The fit params from the MCMC after importance sampling
                  -err_params (array): The errros of the parameters from the MCMC after importance sampling """
    chains = np.loadtxt(file)
    params = chains[burnin:,]
    fit_params = np.zeros(6)
    err_params = np.zeros(6)
    new_params = params.copy()
    for i in range(6):
        new_params[:,i] =gaussian(params[:,3],0.0544,0.0073, len(params[:,3]))*params[:,i]
        plt.show()
        fit_params[i] = new_params[:,i].mean()
        err_params[i] = new_params[:,i].std()/np.sqrt(len(params[:,3]))
    return fit_params, err_params


#Make the plot for question 3 and 4
pars, err = plot_chains(3,'chains_Question3.txt', FFT=True)
pars2, err2 = plot_chains(4,'chains_Question4.txt')
#Get the parameters and error after the prior sampling
prior_fit, prior_err = prior_sample('chains_Question3.txt')

#Write the final results in the summary file
file = open('Final_result.txt', 'a')
file.write('QUESTION 3\n')
file.write('-----------\n')
file.write('The MCMC method gives the following method:\n')
file.write(f'\t-H0    = {pars[0]} pm {err[0]}\n')
file.write(f'\t-ombh2 = {pars[1]} pm {err[1]}\n')
file.write(f'\t-omch2 = {pars[2]} pm {err[2]}\n')
file.write(f'\t-tau   = {pars[3]} pm {err[3]}\n')
file.write(f'\t-As    = {pars[4]} pm {err[4]}\n')
file.write(f'\t-ns    = {pars[5]} pm {err[5]}\n\n')
file.write(f'We see that our chains have converged since we both the chains and the FFT of the chains look like white noise,\n')
file.write(f'as one can see from the files Question3_chains.pdf and Question3_chains_FFT.pdf.\n\n\n')


file.write('QUESTION 4\n')
file.write('-----------\n')
file.write('The MCMC method when constraining the value of tau gives the following method:\n')
file.write(f'\t-H0    = {pars2[0]} pm {err2[0]}\n')
file.write(f'\t-ombh2 = {pars2[1]} pm {err2[1]}\n')
file.write(f'\t-omch2 = {pars2[2]} pm {err2[2]}\n')
file.write(f'\t-tau   = {pars2[3]} pm {err2[3]}\n')
file.write(f'\t-As    = {pars2[4]} pm {err2[4]}\n')
file.write(f'\t-ns    = {pars2[5]} pm {err2[5]}\n\n')
file.write('Note the we constraint sigma within 3-sigma since that includes 99.7% of the expected measurements\n\n')



file.write('The MCMC method when importance sample by a gaussian our result for question 3 give:\n')
file.write(f'\t-H0    = {prior_fit[0]} pm {prior_err[0]}\n')
file.write(f'\t-ombh2 = {prior_fit[1]} pm {prior_err[1]}\n')
file.write(f'\t-omch2 = {prior_fit[2]} pm {prior_err[2]}\n')
file.write(f'\t-tau   = {prior_fit[3]} pm {prior_err[3]}\n')
file.write(f'\t-As    = {prior_fit[4]} pm {prior_err[4]}\n')
file.write(f'\t-ns    = {prior_fit[5]} pm {prior_err[5]}\n\n')
file.write('We see that the to results are really similar, and thus both methods are equivalent.\n')
file.write('We also see that the error is smaller for the importance sampling.')



