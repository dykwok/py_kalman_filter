import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

delta_t = 0.1
t = np.arange(0,5,delta_t)
N = len(t)
sz = (2,N)
g = 10
x = 1/2*g*t**2
Z = np.random.normal(x,5)
Q = np.mat([[0,0],[0,0.9]])
R = 10

A = np.mat([[1,delta_t],[0,1]])
B=np.mat([1/2*delta_t**2,delta_t]).T
H=np.mat([1,0])

n = np.shape(Q)

x_Hat=np.mat(np.zeros(sz))      # a posteri estimate of x
# P=np.mat(np.zeros(n))         # a posteri error estimate
P=[[2,0],[2,0]]                 # a posteri error estimate
x_HatMinus=np.mat(np.zeros(sz)) # a priori estimate of x
P_Minus=np.mat(np.zeros(n))     # a priori error estimate
K=np.mat([[0],[0]])             # gain or blending factor
I = np.mat(np.eye(2))

statistics = np.zeros(N)
for k in range(9,N):
    # time update
    x_HatMinus[:,k] = A*x_Hat[:,k-1]+ B*g
    P_Minus = A*P*A.T+Q
    # measurement update
    K = P_Minus*H.T/(H*P_Minus*H.T+R)
    x_Hat[:,k] = x_HatMinus[:,k]+K*(Z[k]-H*x_HatMinus[:,k])
    P = (I-K*H)*P_Minus

    # data statistics
    statistics[k] = x_Hat[0,k]-Z[k]

plt.subplot(3,1,1)
plt.plot(t,Z,'r--',label = 'Z')
plt.plot(t,x,'-.', label = 'x')
plt.plot(t,x_Hat[0,:].T, label = 'kf')
plt.legend()

plt.subplot(3,1,2)
plt.plot(t,statistics, label = 'error')
plt.legend()

# the histogram of the data
plt.subplot(3,1,3)
n, bins, patches = plt.hist(statistics,30,normed=1,label='hist')

# add a 'best fit' line
mu = np.average(statistics)
sigma = np.std(statistics)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, '--',label='best fit')
plt.legend()
print('mu:',mu,'sigma:',sigma)
print('1 sigma:',mu-sigma,mu+sigma)
print('2 sigma:',mu-sigma*2,mu+sigma*2)
print('3 sigma:',mu-sigma*3,mu+sigma*3)

plt.show()

