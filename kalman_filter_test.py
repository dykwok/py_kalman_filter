import numpy as np
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
P=[[2,0],[2,0]]         # a posteri error estimate
x_HatMinus=np.mat(np.zeros(sz)) # a priori estimate of x
P_Minus=np.mat(np.zeros(n))    # a priori error estimate
K=np.mat([[0],[0]])         # gain or blending factor
I = np.mat(np.eye(2))

for k in range(9,N):
    # time update
    x_HatMinus[:,k] = A*x_Hat[:,k-1]+ B*g
    P_Minus = A*P*A.T+Q
    # measurement update
    K = P_Minus*H.T/(H*P_Minus*H.T+R)
    x_Hat[:,k] = x_HatMinus[:,k]+K*(Z[k]-H*x_HatMinus[:,k])
    P = (I-K*H)*P_Minus

plt.plot(t,Z)
plt.plot(t,x_Hat[0,:].T)
plt.plot(t,x)
plt.show()
