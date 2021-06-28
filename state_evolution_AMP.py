import numpy as np
from scipy import special
import matplotlib.pyplot as plt

p0 = 0.7 #probability of zero element
delta = 0.7 #undersampling fraction (or indeterminacy)
lam = 1.0
itermax = 50

def psi(sigma):
    return -(-np.sqrt(np.pi)*delta**(3/2)*(p0 - 1)*(np.sqrt(2)*np.sqrt(np.pi)*delta**(7/2)*sigma*(lam*sigma - 1)*np.exp(delta*(lam*sigma - 1)**2/(2*sigma**2)) + np.sqrt(2)*np.sqrt(np.pi)*delta**(7/2)*sigma*(lam*sigma + 1)*np.exp(delta*(lam*sigma + 1)**2/(2*sigma**2)) - np.pi*delta**4*(special.erf(np.sqrt(2)*np.sqrt(delta)*(lam*sigma - 1)/(2*sigma)) + special.erf(np.sqrt(2)*np.sqrt(delta)*(lam*sigma + 1)/(2*sigma)))*np.exp(delta*((lam*sigma - 1)**2 + (lam*sigma + 1)**2)/(2*sigma**2)) + 1.0*np.pi*delta**3*sigma**2*(delta*lam**2 + 1)*(special.erf(np.sqrt(2)*np.sqrt(delta)*(lam*sigma - 1)/(2*sigma)) + special.erf(np.sqrt(2)*np.sqrt(delta)*(lam*sigma + 1)/(2*sigma)) - 2)*np.exp(delta*((lam*sigma - 1)**2 + (lam*sigma + 1)**2)/(2*sigma**2)))*np.exp(delta*lam**2/2)/2 + np.pi*delta**4*p0*sigma**2*(1.0*np.sqrt(np.pi)*np.sqrt(delta)*(delta*lam**2 + 1)*(special.erf(np.sqrt(2)*np.sqrt(delta)*lam/2) - 1)*np.exp(delta*lam**2/2) + np.sqrt(2)*delta*lam)*np.exp(delta*((lam*sigma - 1)**2 + (lam*sigma + 1)**2)/(2*sigma**2)))*np.exp(-delta*(lam**2*sigma**2 + (lam*sigma - 1)**2 + (lam*sigma + 1)**2)/(2*sigma**2))/(np.pi**(3/2)*delta**(11/2))

#threshoulding parameter
#estimated variance of errors
sigma = 1.0

x = np.arange(0.01,5.0,0.1)
y = psi(np.sqrt(x))
plt.plot(x,y)
plt.plot(y,x)
plt.xlabel("$\sigma^2$")
plt.ylabel("$\sigma^2$")
plt.show()
plt.close()


#start main algo.
#for i in range(itermax): #iteration count
#    result = "{0}, {1}".format(i+1,sigma)
#    print(result)
#    sigma = psi(np.sqrt(sigma))
