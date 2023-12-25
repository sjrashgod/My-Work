import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.linear_model import LinearRegression

def Z(V,T):
    k = 1.38 * 10 ** (-23)
    m = 3.32 * 10 ** (-27)
    h = 6.62 * 10 ** (-34)
    func=lambda x: (np.pi/2)*(x**2)* (np.e**((-(x**2)*(h**2))/(8*m*(V**(2/3))*k*T)))
    val = integrate.quad(func, 0, 10 ** (11))
    return val

T=np.linspace(150,450,50)
V=np.linspace(20*10**(-3),50*10**(-3),5)

M=[]
for i in V:
    L=[]
    for j in T:
        L.append(Z(i,j)[0])
        #print(Z(i,j))
    plt.plot(T,np.log(np.array(L)))
    M.append(np.array(L))
plt.grid()
plt.show()

M=np.array(M).T
print(M)

A=[]
for i in range(len(T)):
    k = 1.38 * 10 ** (-23)
    Z_diff= np.log(M[i,:][:-1]) - np.log(M[i,:][1:])
    V_diff= V[:-1] - V[1:]
    P= 6.022* 10**(23) * k* i* (Z_diff/V_diff)
    plt.plot(V[:-1],P)
    A.append(P)
plt.grid()
plt.show()
A=np.array(A).T
print(A)

for i in range(len(V_diff)):
    k = 1.38 * 10 ** (-23)
    P=A[i,:]
    #print(len(P))
    plt.plot(T[1:],P[1:])
plt.grid()
plt.show()


for i in range(len(V)):
    k = 1.38 * 10 ** (-23)
    Z_diff= np.log(M[:,i][:-1]) - np.log(M[:,i][1:])
    T_diff= T[:-1] - T[1:]
    E= k* (i**2) * (Z_diff/T_diff)
    plt.plot(T[:-1],E)
    model = LinearRegression().fit(T_diff.reshape(-1, 1), E)
    print("Slope (Cv):", model.coef_)

plt.title('Energy VS Temperature')
plt.grid()
plt.show()

