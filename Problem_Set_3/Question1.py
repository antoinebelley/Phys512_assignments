#Question 1 of problem set 3 in phys 512
#We want to find the chi_squared for the fit done by
#the prof. Since CAMB returns an array rather than a function
#we interpolate between the point of the array and evaluate the function
#at the x-value of the data from wmap
#Author : Antoine Belley
#Date:    17/10/2019
#WARNING: needs python 3.6 or higher due to the use of f-strings


import numpy as np
from scipy.interpolate import interp1d
from wmap_camb_example import cmb, wmap


def chi_squared(data, fit, error):
    """Function to get the chi_squared of the data from
    the data and the fit.
    Parameters: data        (array): y values of the data
                fit         (array): y values of the fit
    Returns:    chi-squared (float): the chi-squared of the fit
    """
    if len(data) != len(fit):
        print("Data and fit don't have the same length... Cannot perform chi-squred...")
        print("Exiting...")
        exit(1)
    else:
        data = np.array(data)
        fit  = np.array(fit)
        chi_squared = np.sum((data-fit)**2/error**2)
        return chi_squared 



x = np.array(wmap[:,0])
model = cmb[2:len(x)+2]
data = wmap[:,1]
error = np.array(wmap[:,2])


f = open('Final_result.txt', 'w')
f.write('QUESTION 1\n')
f.write('-----------\n')
f.write(f'The chi_squared is {chi_squared(data,model, error)}.\n\n')
f.close()




