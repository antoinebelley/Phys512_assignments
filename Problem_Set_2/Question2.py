#Fitting of the data of a stellar flare using Newton's method
#Author: Antoine Belley
#Date: 29/09/2019

import numpy as np
import matplotlib.pyplot as plt

#Import the data 
data = np.loadtxt('229614158_PDCSAP_SC6.txt',delimiter=',')

x=data[:,0][3200:3300]
y=data[:,1][3200:3300]


#####################################################Part a)####################################################
print('Part a)')


def general_exp(x,p):
    """Return the general form of an exponential.
    -Arguments: -x       (array): The point at which we need to evaluate the function
                -p       (array): The paramters for the exponential in the order [x0,a,b,c]
    -Returns:   -func (function): The value of the exp of the form exp(-(x-x0)/b)+c evaluated at the x-values"""
    return np.exp(-(x-p[0])/p[1])+p[2]


#Starting guess values for parameters of the exponenial
x0=data[np.where(data[:,1] == np.max(data[:,1]))][0][0] #Point where the flare starts
a=data[np.where(data[:,1] == np.max(data[:,1]))][0][1]-1 #Amplitude
c=1 #Background line
b=0.02 #Found by trial and error
x0 = x0+np.log(a)*b #Value of x0 when we include the amplitude of the flare
p=np.array([x0,b,c])


#Prints answer to question 2a):
print("The model would look like f(x) = e^(-(x-x0)/b)+c.")
print("This model is obviously not linear since we are dealing with an exponential function...")
print(f"To find the value of x0, we consider the amplitude of the flare a. Converting it into an exponential")
print(f"We get that x0 = t0+ln(a)*b = {x0}, where t0 is the begining of the flare.")
print(f" and c ~ {c} since the background line is at one.")
print(f"We can use trial and error to find b ~ {b}.")


#Plot the guess and values for time arround the flare
guess = general_exp(x, p)
plt.plot(x,y)
plt.plot(x, guess)
plt.show()


########################################Part b)##############################################################
print('\nPart b)')


def grad_exp(x,p):
    """Return gradient of the general form of an exponential.
    -Arguments: -x     (array): The point at which we need to evaluate the function
                -p     (array): The paramters for the exponential in the order [x0,a,b,c]
    -Returns:   -guess (array): The value of the exp of the form a*exp(-(x-x0)/b)+c evaluated at the x-values
                -grad  (array): Gradient of the fucntion (in the same order as the input parameters"""
    guess = general_exp(x,p)
    grad=np.zeros([x.size,p.size]) #Since there is for parameters. there will be 4 derivatives
    grad[:,0] = 1/p[1]*np.exp(-(x-p[0])/p[1]) #Derivative with respect to x0
    grad[:,1] = 1/p[1]**2*(x-p[0])*np.exp(-(x-p[0])/p[1]) #Derivative with respect to b
    grad[:,2] = 1 #Derivative with respect to c
    return guess, grad


def newton_method(data,x,p):
    """Performs Newton's method to find the best fit. Perform 5 iterations. If chi_squared < 1 and 
    reduces by less than 0.1% of its value, stops the iteration.
    -Arguments: -data            (array): Data point to fit
                -x               (array): The x-value at which the function is evaluated
                -p               (array): The paramters for the exponential in the order [x0,a,b,c]
    -Returns:   -p               (array): The  updated paramters for the exponential in the order [x0,a,b,c]
                -chi_squared_new (array): The chi-square of the best fit"""
    chi_squared = 1000
    for j in range(5):
        guess,grad=grad_exp(x,p)
        r=data-guess
        chi_squared_new=(r**2).sum()
        r=np.matrix(r).transpose()
        grad=np.matrix(grad)
        lhs=grad.transpose()*grad
        rhs=grad.transpose()*r
        dp=np.linalg.inv(lhs)*(rhs)
        for jj in range(p.size):
            p[jj]=p[jj]+dp[jj]
        #Stop iterating if the chi-square is small enough 
        if chi_squared_new < 1 and  chi_squared_new - chi_squared/chi_squared < 1 and j>0:
            break
        else:
            chi_squared = chi_squared_new
    return p,chi_squared_new

p_new,chi_squared= newton_method(y,x,p)
plt.plot(x,y, label = 'Data')
plt.plot(x,guess, label = 'Initial guess')
plt.plot(x,general_exp(x,p_new), label = 'Newton\'s method fit')
plt.legend()
plt.show()

print(f'The optimal parameters are x0 = {p_new[0]}, b = {p_new[1]} and c = {p_new[2]}')



#################################Part c)#########################################
print('\nPart c)')

def error_fit(data, x, p):
    """Estimates the error on the parameters using the covariance matrix
    and taking the square-root of its diagonal
    -Arguments: -data (array): Data point to fit
                -x    (array): The x-value at which the function is evaluated
                -p    (array): The paramters for the exponential in the order [x0,a,b,c]
    -Returns:   -perr (array): The error on the parameters"""
    p,chi_squared = newton_method(data,x, p)
    guess, grad = grad_exp(x,p)
    noise = np.diag(1/((data-guess)*(data-guess)))
    pcov = np.dot(grad.transpose(),noise)
    pcov = np.dot(pcov, grad)
    pcov = np.linalg.inv(pcov)
    perr = np.sqrt(np.diag(pcov))
    return  perr, pcov

perr,pcov = error_fit(y,x,p)

print(f'The error on x0 is {perr[0]}, on b is {perr[1]} and on c is {perr[2]}')



#################################Part d)#########################################
print('\nPart d)')

print('I wouldn\'t trust the errors too much since there seems to be')
print('some correlation in the noise of the data. We should thus change our N matrix')
print('to take that into consideration. However, it is still fair to assume that the error')
print('wouldn\'t increase that much.')


