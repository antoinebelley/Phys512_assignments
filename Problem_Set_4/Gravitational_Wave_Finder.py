#Defintion of the fucntion to find the gravitational waves
#Requires python 3.6 or higher (because of the use of f-strings)
#Author: Antoine Belley
#12/11/19

from util import *
import matplotlib.pyplot as plt
import os

def grav_wave_finder(event_name, strain_H, strain_L, th, tl, window, dt):
    """Fucntion to find the gravtiational waves siganl from the LIGO data
       -Arguments: -event_name(string): The name of the event
                   -strains_H (array) : The data coming from the Hanford observatory
                   -strains_L (array) : The data coming from the Livingston observatory
                   -th        (array) : The template from the Hanford observatory
                   -tl        (array) : The template from the Livingston observatory
    """
    
    os.makedirs(event_name, exist_ok=True)
    os.chdir(event_name)

    #Create an array for the time domain and frequency domain for plotting purposes
    time = np.linspace(0,len(strain_H)/4096,len(strain_H))
    frequency = np.fft.rfftfreq(len(strain_H),dt)

    #Generate the noise models for the two observatories
    noise_H, norm_H = noise(strain_H, window)
    noise_L, norm_L = noise(strain_L, window)
    #Plot the power spectrum from the event in log log scale
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle(f'Power Spectrum for {event_name}')
    axs[0].semilogy(frequency, noise_H)
    axs[1].semilogy(frequency, noise_L)
    axs[0].set_ylabel('Hanford Power Spectra')
    axs[1].set_ylabel('Livingston Power Spectra') 
    plt.xlabel('Freq (Hz)')
    plt.savefig(f'Noise_model_{event_name}')

    #Compute the whitened A and d for both observatories
    wA_H = whitenedA(th,noise_H, window, norm_H)
    wA_L = whitenedA(tl,noise_L, window, norm_L)
    wd_H = whitenedd(strain_H,noise_H, window, norm_H)
    wd_L = whitenedd(strain_L,noise_L, window, norm_L)
    #Generate the matched_filter and SNR for the event

    m_H = matched_filter(wA_H, wd_H)
    m_L = matched_filter(wA_L, wd_L)
    #Plot the matched filters from the event
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle(f'Mathed Filters for {event_name}')
    axs[0].plot(time, m_H)
    axs[1].plot(time, m_L)
    axs[0].set_ylabel('Hanford Matched Filter')
    axs[1].set_ylabel('Livingston Matched Filter') 
    plt.xlabel('Time (s)')
    plt.savefig(f'Matched_filters_{event_name}')

    
    #Find the SNR for the two observatories
    SNR_H    = SNR(m_H, wA_H)
    SNR_L    = SNR(m_L, wA_L)
    SNR_comb = SNR_combined(SNR_H, SNR_L)
    #Plot the SNR for the event for each observaotries and both together
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle(f'SNR for {event_name}')
    axs[0].plot(time, SNR_H)
    axs[1].plot(time, SNR_L)
    axs[2].plot(time, SNR_comb)
    axs[0].set_ylabel('Hanford SNR')
    axs[1].set_ylabel('Livingston SNR')
    axs[2].set_ylabel('Combined SNR') 
    plt.xlabel('Time (s)')
    plt.savefig(f'SNR_{event_name}')

    #Find the SNR for the two observatories
    SNR_analytic_H = SNR_analytic(wA_H)
    SNR_analytic_L = SNR_analytic(wA_L)
    SNR_comb_Analytic = SNR_combined(SNR_analytic_H, SNR_analytic_L)
    #Plot the SNR for the event for each observaotries and both together
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle(f'SNR for {event_name}')
    axs[0].plot(time, SNR_H)
    axs[1].plot(time, SNR_L)
    axs[2].plot(time, SNR_comb_Analytic)
    axs[0].set_ylabel('Hanford \n Analytic SNR')
    axs[1].set_ylabel('Livingston \n  Analytic SNR')
    axs[2].set_ylabel('Combined \n Analytic SNR') 
    plt.xlabel('Time (s)')
    plt.savefig(f'Analytic_SNR_{event_name}')


    #Find the frequency where where we have half of the power
    freq_H = freq_half(wA_H, frequency)
    freq_L = freq_half(wA_L, frequency)


    #Find the time of arrival and the error on it for both observaotries
    time_H, err_H = arrival_time(SNR_H, time)
    time_L, err_L = arrival_time(SNR_L, time)

    diff = np.abs(time_H-time_L)
    dist = 5e6
    err_tot = diff*3e8/dist
    os.chdir('..')

    results = { 'name'   : event_name,
                'freq_H' : freq_H,
                'freq_L' : freq_L,
                'time_H' : time_H,
                'err_H'  : err_H,
                'time_L' : time_L,
                'err_L'  : err_L,
                'err_tot': err_tot}

    return results