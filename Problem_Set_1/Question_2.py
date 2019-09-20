#Interpolates the value of the temparature of the Lakeshore diode 
#using cubic interpolation.
#Author: Antoine Belley
#14/09/19
#Requires python 3.6 or higher due to the use of f-string

from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Read the file 'lakeshore.txt' using pandas which contains the data on which we want to interpolate
df = pd.read_csv('lakeshore.txt', delim_whitespace=True, header=None)
columns = ['T', 'V', 'deriv']
df.columns = columns
df = df[::-1] #reindex the data set so that we have the voltage in increasing order

#The derivative is given in mV/K so we need to mutiply the value by 0.001 to keep
#the untis consitstent
df['deriv'] = (1/(0.001*df['deriv']))

#We use scipy.interpolate to interpolate a cubic 
f = interp1d(df['V'], df['T'], kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')
#We use extrapolate only to be able to take the derivative on the full range
#This is in no way a good way to extrapolate the data...


#Generate an array spanning the range of the values of T with 1000 points
x = np.linspace(df['V'][len(df['V'])-1], df['V'].tail(1), 1000)


def interpolation(V):
    """Print the interpolated voltage for a certain T
    -Arguments: -T (float) Temperature at which we want to interpolate"""
    print(f'The temperature at {V} volts is of {f(V)} pm {error_cubic(V)} Kelvin.')


def plot_interpolation(x):
    """Plot the interpolation to insure us that it make sense.
    Arguments: x (array) Range of x on which we want to plot the interpolation"""
    data = []
    for i in x:
        y = f(i)
        data.append(y)
    plt.clf()
    plt.plot(x, data, label = 'Interpolation')
    plt.plot(df['V'], df['T'], marker='*', linestyle = '', label= 'Data points')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature response of the DT-670 from Lakeshore')
    plt.legend()
    plt.show()
     

def error_cubic(V):
    """Calculate the error of the interpolation at a certain point
    Arguments: - V (float) Value at which we want to find the answer
    Returns:   - error_dx (float) Error at V"""
    def derivative(function, x, dx):
        """Calculate the derivative of a fucntion using the higher-order 
        shceme detailed in Question_1.pdf
        Arguments: -fucntion : (function)  The function to be derivated
                   -x        : (float)     Point at wich the function is being derivated
                   -dx       : (float)     Step-size for the derivative
        Returns:   -fp       : (float)     Derivatives of the point evaluated at x"""
        fp = 8.0*(function(x+dx)-function(x-dx)) - (function(x+2.0*dx)-function(x-2.0*dx))
        fp /= 12.0*dx
        return fp
    # To estimate the error we compare the derivative of our function to the one of our interpolation
    # We then find the distance from the closest point and multiply it to the difference of the two derrivatives
    deriv = []
    steps = []
    for i in df['V']:    
        fp = derivative(f,i,1e-4)
        deriv.append(fp)
    closest_V = df.iloc[(df['V'][1:-1]-V).abs().argsort()[:1]]
    closest_V =float(closest_V['V'])
    index_V = df['V'][1:-1][df['V']==closest_V].index[0]
    deriv = np.array(deriv)
    error_dx = np.abs(deriv-df['deriv'])[index_V]*np.absolute(closest_V-V)
    return error_dx


plot_interpolation(x)
interpolation (0.5)

