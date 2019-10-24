#Question 3 of problem set 3 in phys 512
#Reads of the results chains and plots the chains and their FFT to show their convergence
#Author : Antoine Belley
#Date:    17/10/2019
#WARNING: needs python 3.6 or higher due to the use of f-strings


import numpy as np
import matplotlib.pyplot as plt


#We import the chains and remove the first 1000 points to remove the burnin


def plot_chains(Question,file, burnin=500):

    chains = np.loadtxt(file)
    params = chains[burnin:,]

    H0    = chains[:,0]
    ombh2 = chains[:,1]
    omch2 = chains[:,2]
    tau   = chains[:,3]
    As    = chains[:,4]
    ns    = chains[:,5]
    chisq = chains[:,6]


    fit_params=np.mean(params,axis=0)

    plt.clf()
    plt.plot(H0)
    plt.axvspan(0,burnin,facecolor='r', alpha=0.5, label='Burn-in')
    plt.xlabel('Steps')
    plt.ylabel('H0')
    plt.xlim(0,5000)
    plt.axhline(fit_params[0], label=f'Average value = {fit_params[0]}', color='orange')
    plt.legend()
    plt.savefig(f'Question{Question}_chain_H0.pdf')

    plt.clf()
    plt.plot(ombh2)
    plt.axvspan(0,burnin,facecolor='r', alpha=0.5, label='Burn-in')
    plt.xlabel('Steps')
    plt.ylabel('ombh2')
    plt.xlim(0,5000)
    plt.axhline(fit_params[1], label=f'Average value = {fit_params[1]}', color='orange')
    plt.legend()
    plt.savefig(f'Question{Question}_chain_ombh2.pdf')

    plt.clf()
    plt.plot(omch2)
    plt.axvspan(0,burnin,facecolor='r', alpha=0.5, label='Burn-in')
    plt.xlabel('Steps')
    plt.ylabel('omch2')
    plt.xlim(0,5000)
    plt.axhline(fit_params[2], label=f'Average value = {fit_params[2]}', color='orange')
    plt.legend()
    plt.savefig(f'Question{Question}_chain_omch2.pdf')

    plt.clf()
    plt.plot(tau)
    plt.axvspan(0,burnin,facecolor='r', alpha=0.5, label='Burn-in')
    plt.xlabel('Steps')
    plt.ylabel('tau')
    plt.xlim(0,5000)
    plt.axhline(fit_params[3], label=f'Average value = {fit_params[3]}', color='orange')
    plt.legend()
    plt.savefig(f'Question{Question}_chain_tau.pdf')

    plt.clf()
    plt.plot(As)
    plt.axvspan(0,burnin,facecolor='r', alpha=0.5, label='Burn-in')
    plt.xlabel('Steps')
    plt.ylabel('As')
    plt.xlim(0,5000)
    plt.axhline(fit_params[4], label=f'Average value = {fit_params[4]}', color='orange')
    plt.legend()
    plt.savefig(f'Question{Question}_chain_As.pdf')

    plt.clf()
    plt.plot(ns)
    plt.axvspan(0,burnin,facecolor='r', alpha=0.5, label='Burn-in')
    plt.xlabel('Steps')
    plt.ylabel('ns')
    plt.xlim(0,5000)
    plt.axhline(fit_params[5], label=f'Average value = {fit_params[5]}',color='orange')
    plt.legend()
    plt.savefig(f'Question{Question}_chain_ns.pdf')

    plt.clf()
    plt.plot(chisq)
    plt.axvspan(0,burnin,facecolor='r', alpha=0.5, label='Burn-in')
    plt.xlabel('Steps')
    plt.ylabel('Chi-squared')
    plt.xlim(0,5000)
    plt.axhline(fit_params[6], label=f'Average value = {fit_params[6]}',color='orange')
    plt.legend()
    plt.savefig(f'Question{Question}_chain_chisq.pdf')

    return fit_params

def plot_fft_chains(Question, file, burnin=500):
    chains = np.loadtxt(file)
    H0    = np.fft.fft(chains[1:5000,0])
    ombh2 = np.fft.fft(chains[1:5000,1])
    omch2 = np.fft.fft(chains[1:5000,2])
    tau   = np.fft.fft(chains[1:5000,3])
    As    = np.fft.fft(chains[1:5000,4])
    ns    = np.fft.fft(chains[1:5000,5])
    plt.clf()
    plt.plot(H0)
    plt.xlabel('Steps')
    plt.ylabel('FFT of H0')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0,5000)
    plt.savefig(f'Question{Question}_chain_FFT_H0.pdf')

    plt.clf()
    plt.plot(ombh2)
    plt.xlabel('Steps')
    plt.ylabel('FFT ofombh2')
    plt.xlim(1,5000)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'Question{Question}_chain_FFT_ombh2.pdf')

    plt.clf()
    plt.plot(omch2)
    plt.xlabel('Steps')
    plt.ylabel('FFT of omch2')
    plt.xlim(1,5000)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'Question{Question}_chain_FFT_omch2.pdf')

    plt.clf()
    plt.plot(tau)
    plt.xlabel('Steps')
    plt.ylabel('FFT of tau')
    plt.xlim(1,5000)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'Question{Question}_chain_FFT_tau.pdf')

    plt.clf()
    plt.plot(As)
    plt.xlabel('Steps')
    plt.ylabel('FFT of As')
    plt.xlim(1,5000)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'Question{Question}_chain_FFT_As.pdf')

    plt.clf()
    plt.plot(ns)
    plt.xlabel('Steps')
    plt.ylabel('FFT of ns')
    plt.xlim(1,5000)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'Question{Question}_chain_FFT_ns.pdf')

pars = plot_chains(3,'chains_Question3.txt')
pars2 = plot_chains(4,'chains_Question4.txt')
plot_fft_chains(3, 'chains_Question3.txt')
file = open('Final_result.txt', 'a')
file.write('QUESTION 3\n')
file.write('-----------\n')
file.write('The MCMC method gives the following method:\n')
file.write(f'\t-chisq = {pars[6]}\n')
file.write(f'\t-H0    = {pars[0]}\n')
file.write(f'\t-ombh2 = {pars[1]}\n')
file.write(f'\t-omch2 = {pars[2]}\n')
file.write(f'\t-tau   = {pars[3]}\n')
file.write(f'\t-As    = {pars[4]}\n')
file.write(f'\t-ns    = {pars[5]}\n\n')
