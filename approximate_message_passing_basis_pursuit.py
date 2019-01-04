#Approximate Message Passing (AMP) for Matching Pursuit
#"How to Design Message Passing Algorithm for Compressed Sensing"
#David L. Donoho, et al.
#Code by K. Ishibashi, 2017.
import numpy as np
import matplotlib.pyplot as plt
N = 10000 #number of elements to estimate
n = 5000 #number of observations (n should be even less than N)
k = 2 #number of nonzero elements
delta = n/N #undersampling fraction (or indeterminacy)
rho = k/n #measure of sparsity
itermax = 30 #maximum number of iteration
accuracy = 1e-10 #parameter to stop iteration

#Show simulation setup
str = """\
N : {0}, n: {1}, k: {2}
Undersampling fraction: {3}, Sparsity:{4}
""".format(N,n,k,delta,rho)
print(str)

#Calculation of MSE
def mean_squared_error(y, t):
    return np.mean((y-t)**2)

#Soft threshould function
def eta(u, x):
    return(np.sign(u)*np.maximum(0,np.absolute(u) - x))

#Derivative of soft threshold function
def d_eta(u, x):
    return np.piecewise(u,[np.absolute(u)<x,np.absolute(u)>=x],[0.0,1.0])

#for plot
#plotx = np.arange(1,itermax+1,1)
#ploty = np.zeros((itermax,1))

#N signals w/ k nonzero elements
x0 = np.zeros((N,1)) #generate an array w/ N zeros
x0[0:k] = np.random.rand(k,1)  #put k nonzero elements
np.random.shuffle(x0) #shuffle it

#1)Observation matrix (n x N), iid Gaussian w/ mean 0, var 1/n
A = np.random.normal(0, np.sqrt(1.0/n), (n,N))

#2)Observation matrix (n x N) (Donoho's model)
#A = 1/np.sqrt(n) * (1 - 2*np.random.randint(0,2,(n,N)))

#n observation signals
y = np.dot(A,x0)

#Appproximate Message Passing ***************
#Initialization
xp = np.zeros((N,1)) #x(0)
z = y #z(0)
tau = 1.0/np.sqrt(n) #tau(0) (depending on channel parameter)
# Note: This free-parameter significantly affects the resulting performance.
x = eta(np.dot(A.T,z),tau) #x(1)= z(0),tau(0)

#start main algo.
for i in range(itermax): #iteration count
    #stop if certain accuracy is satisfied
    if tau < accuracy:
        break
    #calculate <\eta^{\prime} (A^* z(t-1) + x(t-1));\tau(t-1)>
    m = d_eta(np.dot(A.T,z) + xp,tau).mean()
    #Factor update LHS: z(x(t), z(t-1), x(t-1), tau(t-1))
    z = y - np.dot(A,x) + np.dot(np.dot(1.0/delta, z),m)
    #update: tau(tau(t-1), z(t-1), x(t-1))
    tau = tau/delta * m
    #hold previous value before update x(t)
    xp = x
    #Variable update rule LHS: x(t+1)
    x = eta(np.dot(A.T,z) + x,tau)
    #Plot MSE at every iteration
    #result = "{0}, {1}".format(i,mean_squared_error(x0,x))
    #print(result)

#show result
print("\nIteration",i,",Resulting MSE:",mean_squared_error(x0,x))
