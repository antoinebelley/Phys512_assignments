#Definiton of the models for the two observatoreies of lIGO
#Requires python 3.6 or higher (because of the use of f-strings)
#Author: Antoine Belley
#12/11/19

from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import numpy as np

def noise(strain, window):
    """Create a noise model for a specific window by considering the power 
    spectrum of the strain (after applying the window), and then apply a gaussian filter to smooth
    the resulting spectrum. The spectrum is given by the square of the FT of the strain. This model represent 
    well the noise since we have a stationary gaussian noise.
    -Arguments: -strain (array): The data for the specific interferometer
                -window (array): The window to insure that we have a 0 value for our strain at the boundary
    -Returns:   -noise  (array): The noise model in our data given by the power spectrum"""
    wstrain = window*strain                        #Apply the chosen window
    norm    = np.sqrt(np.mean(window**2))          #Compute the norm of the window
    noise   = np.abs(np.fft.rfft(wstrain)/norm)**2 #Compute the power spectrum
    noise   = gaussian_filter(noise, 1)            #Convolute the power spectrum with the gaussian filter
    return noise, norm


def whitenedA(template,noise, window, norm):
    """Computes the whitened A matrix, ie in the basis where N == I after applying our window
    -Arguments: -template (array): The template for the signal
                -noise    (array): The noise model for our data
                -window   (array): The window to insure that we have a 0 value for our strain at the boundary
                -norm     (float): The normalization factor of our window
    -Returns:   -wA       (array): The whithened A matrix"""
    return np.fft.rfft(window*template)/(np.sqrt(noise)*norm) #The whitened value of A (I.e. in the basis where N == I)


def whitenedd(strain,noise, window, norm):
    """Computes the whitened d vector, ie in the basis where N == I after applying our window
    -Arguments: -strain (array): The data for the specific interferometer
                -noise  (array): The noise model for our data
                -window (array): The window to insure that we have a 0 value for our strain at the boundary
                -norm   (float): The normalization factor of our window
    -Returns:   -wd     (array): The whithened d vector"""
    return np.fft.rfft(window*strain)/(np.sqrt(noise)*norm)   #The whitened value of d (I.e. in the basis where N == I)


def matched_filter(wA, wd):
    """Use a matched filter to find the value of our model. We are using here the fact that
            m = (IFT(FT(N^0.5*A)^T)*FT(N^0.5*d)))/((N^0.5*A)^T*(N^0.5*A))
    and that N is the identity if we choose the right basis. We also need to apply the window to both our data and 
    the template since we applied it for our noise. 
    -Arguments: -wA (array): The whithened A matrix
                -wd (array): The whithened d vector
    -Returns:   -m  (array): The matched_filter for the event/observattory"""
    #use the match filter on our noise model to find the value of m
    return np.fft.fftshift(np.fft.irfft(np.conj(wA)*wd))

def SNR(m,wA):
    """We  compute the siganl to noise ratio using
            m*sqrt((N^0.5*A)^T*(N^0.5*A))
    -Arguments: -m   (array): The matched_filter for the event/observartory
                -wA  (array): The whithened A matrix
    -Returns:   -SNR (array): The signal to noise ratio event/observartory"""
    return np.abs(m*np.fft.fftshift(np.fft.irfft(np.sqrt(np.conj(wA)*wA))))

def SNR_analytic(wA):
    """We  compute the anlaytic siganl to noise ratio which is simply the whithened A matrix back in the time domain
    -Arguments: -wA  (array): The whithened A matrix
    -Returns:   -SNR_anlaytic (array): The analytic signal to noise ratio event/observartory"""
    return np.abs(np.fft.irfft(wA))


def SNR_combined(SNR_H, SNR_L):
    """Computes the signal to noise ratio from the two observatories together
    -Arguments: -SNR_H    (array): The signal to noise ratio for the Hanford Observatory
                -SNR_L    (array): The template for the signal
    -Returns:   -SNR_comb (array): The signal to noise ratio for the matched filter"""

    return np.sqrt(SNR_H**2+SNR_L**2)


def freq_half(wA, frequency):
    """Computes the signal to noise ratio from the two observatories together
    -Arguments: -wA        (array): The whithened A matrix
                -frequency (array): Frequency domain
    -Returns:   -freq_half (array): The frequency  frequency from each event where half the weight comes from above that frequency and half below"""
    sum_power_spectra = np.cumsum(np.abs(wA**2))
    freq_half=frequency[np.argmin(np.abs(sum_power_spectra-(np.amax(sum_power_spectra)/2)))]
    return freq_half


def gaussian(x,Amp,sig,mean):
    """Gaussian function used to fit the timeof arrival.
    -Arguments: -x           (array): The data to be fitted
                -Amp         (float): The mean of the distribution
                -sig         (float): The standard deviation of the distribution
                -mean        (float): Mean of the distribution
    -Returns:   -guassian (function): Function to be fitted"""
    return Amp*np.exp(-(x-mean)**2/sig**2)


def arrival_time(SNR,x):
    """Compute the arrival time of the signal by fitting a gaussian to the SNR vs time plot. We 
    consider the mean to be the arrival time and the standard deviation to be the error
    -Arguments: - Siganl (array): The signal to noise ratio event/observartory
                - x      (array): The data to be fitted
    -Retruns:   - mean   (float): The time of arrival of the signal
                - err    (float); The error in the data"""
    Amp = np.amax(SNR)
    pos = np.argmax(SNR)
    mean = x[pos]
    sig = 0.001
    pars, cov = curve_fit(gaussian, x[pos-5:pos+5], SNR[pos-5:pos+5], p0=[Amp,sig,mean])
    return pars[2], pars[1]