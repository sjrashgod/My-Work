#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy import stats

def My_RK4(Y0,f,xi,xf,N):
    x0 = np.linspace(xi,xf,N)
    h  = (xf-xi )/(N-1)
    Y = np.zeros((N,len(Y0)))
    k1 = np.zeros((N,len(Y0)))
    k2 = np.zeros((N,len(Y0)))
    k3 = np.zeros((N,len(Y0)))
    k4 = np.zeros((N,len(Y0)))
    k = np.zeros((N,len(Y0)))
    Y[0,:] = Y0
    for j in range(N-1):
        k1[j+1,:] = h*f(x0[j],Y[j,:])
        k2[j+1,:] = h*f(x0[j]+h*0.5,Y[j,:]+k1[j+1,:]*0.5)
        k3[j+1,:] = h*f(x0[j]+h*0.5,Y[j,:]+k2[j+1,:]*0.5)
        k4[j+1,:] = h*f(x0[j]+h,Y[j,:]+k3[j+1,:])
        k[j+1,:] = (k1[j+1,:]+2*k2[j+1,:]+2*k3[j+1,:]+k4[j+1,:])/6
        Y[j+1,:] = Y[j,:]+  k[j+1,:]

    return Y               

def SecantMethod(S_0,S_1,phi_0,phi_1,tol,maxiter):
    for i in range(maxiter):
        S_2 = S_1 - ((S_1-S_0)/(phi_1- phi_0))*phi_1
        if abs(S_2-S_1) < tol:
            break
            return S_2
        else:
            S_0 = S_1
            S_1 = S_2
    return S_2
        


# In[86]:


def Exact(x):
    y = 1/(x+3)
    return y

def Exact_deri(N):
    x = sy.symbols("x",real = True)
    f =  1/(x+3)
    y_deri = sy.diff(f , x)
    y_value = sy.lambdify(x,y_deri)
    x = np.linspace(0,1,N)
    y = y_value(x)
    return y

def func(x,x_vec):
    ans_vec = np.zeros((2))
    ans_vec[0] = x_vec[1]
    ans_vec[1] = 2*x_vec[0]**3
    return ans_vec

def Lin_shooting_D(S0,S1,func,xi,xf,beta,a,Nmax,tol):
    y0 = My_RK4([1/3,S0],func,xi,xf,Nmax)
    y1 = My_RK4([1/3,S1],func,xi,xf,Nmax)
    phi0 = beta - y0.T[a,Nmax-1]
    phi1 = beta - y1.T[a,Nmax-1]
    for i in range(Nmax):
        if abs(phi0)<tol:
            break
            return y0.T[0],phi0
        else:
            S2 = SecantMethod(S0,S1,phi0,phi1,tol,1000)
            S0 = S1
            S1 = S2  
            y3 = My_RK4([1/3,S2],func,xi,xf,Nmax).T[a]
            y4 = My_RK4([1/3,S2],func,xi,xf,Nmax).T[a-1]
            phi2 = beta  - My_RK4([1/3,S2],func,xi,xf,Nmax).T[a,-1]
            phi0 = phi1 
            phi1 = phi2
    return y3,y4
            
y3,y4 = Lin_shooting_D(1,0,func,0,1,0.25,0,4,0.0000000001)
y5,y6 = Lin_shooting_D(1,0,func,0,1,0.25,0,8,0.0000000001)
x1 = np.linspace(0,1,4)
y_exact = Exact(x1)
Error = abs(y_exact - y3)
List = {"x":x1,"y_num":y3,"y_exact":y_exact,"Error":Error}
num = pd.DataFrame(List)
print(num)

x2 = np.linspace(0,1,8)
y_exact1 = Exact(x2)
Error1 = abs(y_exact1 - y5)
List1 = {"x":x2,"y_num":y5,"y_exact":y_exact1,"Error":Error1}
num1 = pd.DataFrame(List1)
print(num1)


# In[103]:


Error = [] ; RMSE = []
print("####################### N = 2 ############################")
y2,y_2 = Lin_shooting_D(1,0,func,0,1,0.25,0,2,0.0000000001)
x2 = np.linspace(0,1,2)
y_exact2 = Exact(x2)
y_exact_D_2 = Exact_deri(2)
result2 = np.sqrt(mean_squared_error(y_exact2,y2))
error2 = max(abs(y_exact2 - y2))
Error.append(error2)
RMSE.append(result2)
List1 = {"x":x2,"y_num":y2,"y_exact":y_exact2}
num1 = pd.DataFrame(List1)
print(num1)

print("####################### N = 4 ############################")
y4,y_4 = Lin_shooting_D(1,0,func,0,1,0.25,0,4,0.0000000001)
x4 = np.linspace(0,1,4)
y_exact4 = Exact(x4)
y_exact_D_4 = Exact_deri(4)
result4 = np.sqrt(mean_squared_error(y_exact4,y4))
error4 = max(abs(y_exact4 - y4))
Error.append(error4)
RMSE.append(result4)
List4 = {"x":x4,"y_num":y4,"y_exact":y_exact4}
num4 = pd.DataFrame(List4)
print(num4)

print("####################### N = 8 ############################")
y8,y_8 = Lin_shooting_D(1,0,func,0,1,0.25,0,8,0.0000000001)
x8 = np.linspace(0,1,8)
y_exact8 = Exact(x8)
y_exact_D_8 = Exact_deri(8)
result8 = np.sqrt(mean_squared_error(y_exact8,y8))
error8 = max(abs(y_exact8 - y8))
Error.append(error8)
RMSE.append(result8)
List8 = {"x":x8,"y_num":y8,"y_exact":y_exact8}
num8 = pd.DataFrame(List8)
print(num8)

print("####################### N = 16 ############################")
y16,y_16 = Lin_shooting_D(1,0,func,0,1,0.25,0,16,0.0000000001)
x16 = np.linspace(0,1,16)
y_exact16 = Exact(x16)
y_exact_D_16 = Exact_deri(16)
result16 = np.sqrt(mean_squared_error(y_exact16,y16))
error16 = max(abs(y_exact16 - y16))
Error.append(error16)
RMSE.append(result16)
List16 = {"x":x16,"y_num":y16,"y_exact":y_exact16}
num16 = pd.DataFrame(List16)
print(num16)

print("####################### N = 32 ############################")
y32,y_32 = Lin_shooting_D(1,0,func,0,1,0.25,0,32,0.0000000001)
x32 = np.linspace(0,1,32)
y_exact32 = Exact(x32)
y_exact_D_32 = Exact_deri(32)
result32 = np.sqrt(mean_squared_error(y_exact32,y32))
error32 = max(abs(y_exact32 - y32))
Error.append(error32)
RMSE.append(result32)
List32 = {"x":x32,"y_num":y32,"y_exact":y_exact32}
num32 = pd.DataFrame(List32)
print(num32)

print("####################### N = 64 ############################")
y64,y_64 = Lin_shooting_D(1,0.5,func,0,1,0.25,0,64,0.0000000001)
x64 = np.linspace(0,1,64)
y_exact64 = Exact(x64)
y_exact_D_64 = Exact_deri(64)
result64 = np.sqrt(mean_squared_error(y_exact64,y64))
error64 = max(abs(y_exact64 - y64))
Error.append(error64)
RMSE.append(result64)
List64 = {"x":x64,"y_num":y64,"y_exact":y_exact64}
num64 = pd.DataFrame(List64)
print(num64)

plt.figure(figsize = (15,10))
display(plt.plot(x2,y2,label = "y_num,N = 2"))
display(plt.scatter(x2,y2))
display(plt.scatter(x2,y_exact2,label = "y_exact,N = 2"))

display(plt.plot(x4,y4,label = "y_num,N = 4"))
display(plt.scatter(x4,y4))
display(plt.scatter(x4,y_exact4,label = "y_exact,N = 4"))

display(plt.plot(x8,y8,label = "y_num,N = 8"))
display(plt.scatter(x8,y8))
display(plt.scatter(x8,y_exact8,label = "y_exact,N = 8"))

display(plt.plot(x16,y16,label = "y_num,N = 16"))
display(plt.scatter(x16,y16))
display(plt.scatter(x16,y_exact16,label = "y_exact,N = 16"))

display(plt.plot(x32,y32,label = "y_num,N = 32"))
display(plt.scatter(x32,y32))
display(plt.scatter(x32,y_exact32,label = "y_exact,N = 32"))

display(plt.plot(x64,y64,label = "y_num,N = 64"))
display(plt.scatter(x64,y64))
display(plt.scatter(x64,y_exact64,label = "y_exact,N = 64"))
display(plt.title("plot Between x and y "))
display(plt.xlabel("x"))
display(plt.ylabel("y"))
display(plt.grid())
display(plt.legend())
display(plt.show())



plt.figure(figsize = (15,10))
display(plt.plot(x2,y_2,label = "y_num,N = 2"))
display(plt.scatter(x2,y_exact_D_2,label = "y_exact,N = 2"))

display(plt.plot(x4,y_4,label = "y_num,N = 4"))
display(plt.scatter(x4,y_exact_D_4,label = "y_exact,N = 4"))

display(plt.plot(x8,y_8,label = "y_num,N = 8"))
display(plt.scatter(x8,y_exact_D_8,label = "y_exact,N = 8"))

display(plt.plot(x16,y_16,label = "y_num,N = 16"))
display(plt.scatter(x16,y_exact_D_16,label = "y_exact,N = 16"))

display(plt.plot(x32,y_32,label = "y_num,N = 32"))
display(plt.scatter(x32,y_exact_D_32,label = "y_exact,N = 32"))

display(plt.plot(x64,y_64,label = "y_num,N = 64"))
display(plt.scatter(x64,y_exact_D_64,label = "y_exact,N = 64"))

display(plt.title("plot Between x and dy/dx "))
display(plt.xlabel("x"))
display(plt.ylabel("dy/dx"))
display(plt.grid())
display(plt.legend())
display(plt.show())

List = {"Error":Error,"RMSE":RMSE}
num = pd.DataFrame(List)
print(num)


# In[108]:


N = [2,4,8,16,32,64]

x= np.array(N).reshape(-1,1)
y=np.array(Error)
model = LinearRegression().fit(x,y)
r_sq = model.score(x,y)
print('intercept:  ' ,model.intercept_)
print('slope :  ' ,model.coef_)

plt.scatter(x,y)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("log(N)")
plt.ylabel("log(E_max)")
plt.title("log(N) vs log(E_max)")
plt.grid()
plt.show()


# In[56]:


def Lin_shooting_N(S0,S1,func,xi,xf,beta,a,Nmax,tol):
    y0 = My_RK4([S0,-1/9],func,xi,xf,Nmax)
    y1 = My_RK4([S1,-1/9],func,xi,xf,Nmax)
    phi0 = beta - y0.T[a,Nmax-1]
    phi1 = beta - y1.T[a,Nmax-1]
    for i in range(Nmax):
        if abs(phi0)<tol:
            break
            return y0.T[0],phi0
        else:
            S2 = SecantMethod(S0,S1,phi0,phi1,tol,1)
            S0 = S1
            S1 = S2  
            y3 = My_RK4([S2,-1/9],func,xi,xf,Nmax).T[a]
            y4 = My_RK4([S2,-1/9],func,xi,xf,Nmax).T[a-1]
            phi2 = beta  - My_RK4([S2,-1/9],func,xi,xf,Nmax).T[a,-1]
            phi0 = phi1 
            phi1 = phi2
    return y3,y4
            


# In[104]:


Error = [] ; RMSE = []
print("####################### N = 2 ############################")
y_2,y2 = Lin_shooting_N(1,0.5,func,0,1,-1/16,1,2,0.0000000001)
x2 = np.linspace(0,1,2)
y_exact2 = Exact(x2)
y_exact_D_2 = Exact_deri(2)
result2 = np.sqrt(mean_squared_error(y_exact2,y2))
error2 = max(abs(y_exact2 - y2))
Error.append(error2)
RMSE.append(result2)
List1 = {"x":x2,"y_num":y2,"y_exact":y_exact2}
num1 = pd.DataFrame(List1)
print(num1)

print("####################### N = 4 ############################")
y_4,y4 = Lin_shooting_N(1,0.5,func,0,1,-1/16,1,4,0.0000000001)
x4 = np.linspace(0,1,4)
y_exact4 = Exact(x4)
y_exact_D_4 = Exact_deri(4)
result4 = np.sqrt(mean_squared_error(y_exact4,y4))
error4 = max(abs(y_exact4 - y4))
Error.append(error4)
RMSE.append(result4)
List4 = {"x":x4,"y_num":y4,"y_exact":y_exact4}
num4 = pd.DataFrame(List4)
print(num4)

print("####################### N = 8 ############################")
y_8,y8 = Lin_shooting_N(1,0.5,func,0,1,-1/16,1,8,0.0000000001)
x8 = np.linspace(0,1,8)
y_exact8 = Exact(x8)
y_exact_D_8 = Exact_deri(8)
result8 = np.sqrt(mean_squared_error(y_exact8,y8))
error8 = max(abs(y_exact8 - y8))
Error.append(error8)
RMSE.append(result8)
List8 = {"x":x8,"y_num":y8,"y_exact":y_exact8}
num8 = pd.DataFrame(List8)
print(num8)

print("####################### N = 16 ############################")
y_16,y16 = Lin_shooting_N(1,0.5,func,0,1,-1/16,1,16,0.0000000001)
x16 = np.linspace(0,1,16)
y_exact16 = Exact(x16)
y_exact_D_16 = Exact_deri(16)
result16 = np.sqrt(mean_squared_error(y_exact16,y16))
error16 = max(abs(y_exact16 - y16))
Error.append(error16)
RMSE.append(result16)
List16 = {"x":x16,"y_num":y16,"y_exact":y_exact16}
num16 = pd.DataFrame(List16)
print(num16)

print("####################### N = 32 ############################")
y_32,y32 = Lin_shooting_N(1,0.5,func,0,1,-1/16,1,32,0.0000000001)
x32 = np.linspace(0,1,32)
y_exact32 = Exact(x32)
y_exact_D_32 = Exact_deri(32)
result32 = np.sqrt(mean_squared_error(y_exact32,y32))
error32 = max(abs(y_exact32 - y32))
Error.append(error32)
RMSE.append(result32)
List32 = {"x":x32,"y_num":y32,"y_exact":y_exact32}
num32 = pd.DataFrame(List32)
print(num32)

print("####################### N = 64 ############################")
y_64,y64 = Lin_shooting_N(1,0.5,func,0,1,-1/16,1,64,0.0000000001)
x64 = np.linspace(0,1,64)
y_exact64 = Exact(x64)
y_exact_D_64 = Exact_deri(64)
print()
result64 = np.sqrt(mean_squared_error(y_exact64,y64))
error64 = max(abs(y_exact64 - y64))
Error.append(error64)
RMSE.append(result64)
List64 = {"x":x64,"y_num":y64,"y_exact":y_exact64}
num64 = pd.DataFrame(List64)
print(num64)

plt.figure(figsize = (15,10))
display(plt.plot(x2,y2,label = "y_num,N = 2"))
display(plt.scatter(x2,y_exact2,label = "y_exact,N = 2"))

display(plt.plot(x4,y4,label = "y_num,N = 4"))
display(plt.scatter(x4,y_exact4,label = "y_exact,N = 4"))

display(plt.plot(x8,y8,label = "y_num,N = 8"))
display(plt.scatter(x8,y_exact8,label = "y_exact,N = 8"))

display(plt.plot(x16,y16,label = "y_num,N = 16"))
display(plt.scatter(x16,y_exact16,label = "y_exact,N = 16"))

display(plt.plot(x32,y32,label = "y_num,N = 32"))
display(plt.scatter(x32,y_exact32,label = "y_exact,N = 32"))

display(plt.plot(x64,y64,label = "y_num,N = 64"))
display(plt.scatter(x64,y_exact64,label = "y_exact,N = 64"))
display(plt.title("plot Between x and y "))
display(plt.xlabel("x"))
display(plt.ylabel("y"))
display(plt.grid())
display(plt.legend())
display(plt.show())



plt.figure(figsize = (15,10))
display(plt.plot(x2,y_2,label = "y_num,N = 2"))
display(plt.scatter(x2,y_exact_D_2,label = "y_exact,N = 2"))

display(plt.plot(x4,y_4,label = "y_num,N = 4"))
display(plt.scatter(x4,y_exact_D_4,label = "y_exact,N = 4"))

display(plt.plot(x8,y_8,label = "y_num,N = 8"))
display(plt.scatter(x8,y_exact_D_8,label = "y_exact,N = 8"))

display(plt.plot(x16,y_16,label = "y_num,N = 16"))
display(plt.scatter(x16,y_exact_D_16,label = "y_exact,N = 16"))

display(plt.plot(x32,y_32,label = "y_num,N = 32"))
display(plt.scatter(x32,y_exact_D_32,label = "y_exact,N = 32"))

display(plt.plot(x64,y_64,label = "y_num,N = 64"))
display(plt.scatter(x64,y_exact_D_64,label = "y_exact,N = 64"))
display(plt.title("plot Between x and dy/dx "))
display(plt.xlabel("x"))
display(plt.ylabel("dy/dx"))
display(plt.grid())
display(plt.legend())
display(plt.show())

List = {"Error":Error,"RMSE":RMSE}
num = pd.DataFrame(List)
print(num)


# In[109]:


N = [2,4,8,16,32,64]

x= np.array(N).reshape(-1,1)
y=np.array(Error)
model = LinearRegression().fit(x,y)
r_sq = model.score(x,y)
print('intercept:  ' ,model.intercept_)
print('slope :  ' ,model.coef_)

plt.scatter(x,y)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("log(N)")
plt.ylabel("log(E_max)")
plt.title("log(N) vs log(E_max)")
plt.grid()
plt.show()


# In[96]:


def Lin_shooting_RD(S0,S1,func,xi,xf,beta,a,Nmax,tol):
    y0 = My_RK4([(2+9*S0)/3,S0],func,xi,xf,Nmax)
    y1 = My_RK4([(2+9*S0)/3,S1],func,xi,xf,Nmax)
    phi0 = beta - y0.T[a,-1]
    phi1 = beta - y1.T[a,-1]
    for i in range(Nmax):
        if abs(phi0)<tol:
            break
            return y0.T[a],phi0
        else:
            S2 = SecantMethod(S0,S1,phi0,phi1,tol,1)
            S0 = S1
            S1 = S2            
            y3 = My_RK4([(2+9*S2)/3,S2],func,xi,xf,Nmax).T[a]
            y4 = My_RK4([(2+9*S2)/3,S2],func,xi,xf,Nmax).T[a-1]
            phi2 = beta  - My_RK4([(2+9*S2)/3,S2],func,xi,xf,Nmax).T[a,-1]
            phi0 = phi1 
            phi1 = phi2
    return y3,y4


# In[105]:


Error = [] ; RMSE = []
print("####################### N = 2 ############################")
y2,y_2 = Lin_shooting_RD(0,0.001,func,0,1,0.25,0,2,0.0000001)
x2 = np.linspace(0,1,2)
y_exact2 = Exact(x2)
y_exact_D_2 = Exact_deri(2)
result2 = np.sqrt(mean_squared_error(y_exact2,y2))
error2 = max(abs(y_exact2 - y2))
Error.append(error2)
RMSE.append(result2)
List1 = {"x":x2,"y_num":y2,"y_exact":y_exact2}
num1 = pd.DataFrame(List1)
print(num1)

print("####################### N = 4 ############################")
y4,y_4 = Lin_shooting_RD(0,1,func,0,1,0.25,0,4,0.0000001)
x4 = np.linspace(0,1,4)
y_exact4 = Exact(x4)
y_exact_D_4 = Exact_deri(4)
result4 = np.sqrt(mean_squared_error(y_exact4,y4))
error4 = max(abs(y_exact4 - y4))
Error.append(error4)
RMSE.append(result4)
List4 = {"x":x4,"y_num":y4,"y_exact":y_exact4}
num4 = pd.DataFrame(List4)
print(num4)

print("####################### N = 8 ############################")
y8,y_8 = Lin_shooting_RD(0,1,func,0,1,0.25,0,8,0.0000001)
x8 = np.linspace(0,1,8)
y_exact8 = Exact(x8)
y_exact_D_8 = Exact_deri(8)
result8 = np.sqrt(mean_squared_error(y_exact8,y8))
error8 = max(abs(y_exact8 - y8))
Error.append(error8)
RMSE.append(result8)
List8 = {"x":x8,"y_num":y8,"y_exact":y_exact8}
num8 = pd.DataFrame(List8)
print(num8)

print("####################### N = 16 ############################")
y16,y_16 =Lin_shooting_RD(0,1,func,0,1,0.25,0,16,0.0000001)
x16 = np.linspace(0,1,16)
y_exact16 = Exact(x16)
y_exact_D_16 = Exact_deri(16)
result16 = np.sqrt(mean_squared_error(y_exact16,y16))
error16 = max(abs(y_exact16 - y16))
Error.append(error16)
RMSE.append(result16)
List16 = {"x":x16,"y_num":y16,"y_exact":y_exact16}
num16 = pd.DataFrame(List16)
print(num16)

print("####################### N = 32 ############################")
y32,y_32 = Lin_shooting_RD(0,1,func,0,1,0.25,0,32,0.0000001)
x32 = np.linspace(0,1,32)
y_exact32 = Exact(x32)
y_exact_D_32 = Exact_deri(32)
result32 = np.sqrt(mean_squared_error(y_exact32,y32))
error32 = max(abs(y_exact32 - y32))
Error.append(error32)
RMSE.append(result32)
List32 = {"x":x32,"y_num":y32,"y_exact":y_exact32}
num32 = pd.DataFrame(List32)
print(num32)

print("####################### N = 64 ############################")
y64,y_64 = Lin_shooting_RD(0,1,func,0,1,0.25,0,64,0.0000001)
x64 = np.linspace(0,1,64)
y_exact64 = Exact(x64)
y_exact_D_64 = Exact_deri(64)
result64 = np.sqrt(mean_squared_error(y_exact64,y64))
error64 = max(abs(y_exact64 - y64))
Error.append(error64)
RMSE.append(result64)
List64 = {"x":x64,"y_num":y64,"y_exact":y_exact64}
num64 = pd.DataFrame(List64)
print(num64)

plt.figure(figsize = (15,10))
display(plt.plot(x2,y2,label = "y_num,N = 2"))
display(plt.scatter(x2,y_exact2,label = "y_exact,N = 2"))

display(plt.plot(x4,y4,label = "y_num,N = 4"))
display(plt.scatter(x4,y_exact4,label = "y_exact,N = 4"))

display(plt.plot(x8,y8,label = "y_num,N = 8"))
display(plt.scatter(x8,y_exact8,label = "y_exact,N = 8"))

display(plt.plot(x16,y16,label = "y_num,N = 16"))
display(plt.scatter(x16,y_exact16,label = "y_exact,N = 16"))

display(plt.plot(x32,y32,label = "y_num,N = 32"))
display(plt.scatter(x32,y_exact32,label = "y_exact,N = 32"))

display(plt.plot(x64,y64,label = "y_num,N = 64"))
display(plt.scatter(x64,y_exact64,label = "y_exact,N = 64"))
display(plt.title("plot Between x and y "))
display(plt.xlabel("x"))
display(plt.ylabel("y"))
display(plt.grid())
display(plt.legend())
display(plt.show())


plt.figure(figsize = (15,10))
display(plt.plot(x2,y_2,label = "y_num,N = 2"))
display(plt.scatter(x2,y_exact_D_2,label = "y_exact,N = 2"))

display(plt.plot(x4,y_4,label = "y_num,N = 4"))
display(plt.scatter(x4,y_exact_D_4,label = "y_exact,N = 4"))

display(plt.plot(x8,y_8,label = "y_num,N = 8"))
display(plt.scatter(x8,y_exact_D_8,label = "y_exact,N = 8"))

display(plt.plot(x16,y_16,label = "y_num,N = 16"))
display(plt.scatter(x16,y_exact_D_16,label = "y_exact,N = 16"))

display(plt.plot(x32,y_32,label = "y_num,N = 32"))
display(plt.scatter(x32,y_exact_D_32,label = "y_exact,N = 32"))

display(plt.plot(x64,y_64,label = "y_num,N = 64"))
display(plt.scatter(x64,y_exact_D_64,label = "y_exact,N = 64"))
display(plt.title("plot Between x and dy/dx "))
display(plt.xlabel("x"))
display(plt.ylabel("dy/dx"))
display(plt.grid())
display(plt.legend())
display(plt.show())



List = {"Error":Error,"RMSE":RMSE}
num = pd.DataFrame(List)
print(num)


# In[111]:


N = [2,4,8,16,32,64]

x= np.array(N).reshape(-1,1)
y=np.array(Error)
model = LinearRegression().fit(x,y)
r_sq = model.score(x,y)
print('intercept:  ' ,model.intercept_)
print('slope :  ' ,model.coef_)

plt.scatter(x,y)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("log(N)")
plt.ylabel("log(E_max)")
plt.title("log(N) vs log(E_max)")
plt.grid()
plt.show()


# In[92]:


def Lin_shooting_DR(S0,S1,func,xi,xf,beta,a,Nmax,tol):
    y0 = My_RK4([1/3,S0],func,xi,xf,Nmax)
    y1 = My_RK4([1/3,S1],func,xi,xf,Nmax)
    phi0 = beta - (2*y0.T[a,-1] + 2*y0.T[a+1,-1])
    phi1 = beta - (2*y1.T[a,-1] + 2*y1.T[a+1,-1])
    for i in range(Nmax):
        if abs(phi0)<tol:
            break
            return y0.T[a],phi0
        else:
            S2 = SecantMethod(S0,S1,phi0,phi1,tol,100)
            S0 = S1
            S1 = S2            
            y3 = My_RK4([1/3,S2],func,xi,xf,Nmax).T[a]
            y4 = My_RK4([1/3,S2],func,xi,xf,Nmax).T[a-1]
            phi2 = beta  - (2*My_RK4([1/3,S2],func,xi,xf,Nmax).T[a,-1] + 2*My_RK4([1/3,S2],func,xi,xf,Nmax).T[a+1,-1])
            phi0 = phi1 
            phi1 = phi2
    return y3,y4


# In[107]:


Error = [] ; RMSE = []
print("####################### N = 2 ############################")
y2,y_2 = Lin_shooting_RD(0,0.0001,func,0,1,0.25,0,2,0.0000001)
x2 = np.linspace(0,1,2)
y_exact2 = Exact(x2)
y_exact_D_2 = Exact_deri(2)
result2 = np.sqrt(mean_squared_error(y_exact2,y2))
error2 = max(abs(y_exact2 - y2))
Error.append(error2)
RMSE.append(result2)
List1 = {"x":x2,"y_num":y2,"y_exact":y_exact2}
num1 = pd.DataFrame(List1)
print(num1)

print("####################### N = 4 ############################")
y4,y_4 = Lin_shooting_RD(0,1,func,0,1,0.25,0,4,0.0000001)
x4 = np.linspace(0,1,4)
y_exact4 = Exact(x4)
y_exact_D_4 = Exact_deri(4)
result4 = np.sqrt(mean_squared_error(y_exact4,y4))
error4 = max(abs(y_exact4 - y4))
Error.append(error4)
RMSE.append(result4)
List4 = {"x":x4,"y_num":y4,"y_exact":y_exact4}
num4 = pd.DataFrame(List4)
print(num4)

print("####################### N = 8 ############################")
y8,y_8 = Lin_shooting_RD(0,1,func,0,1,0.25,0,8,0.0000001)
x8 = np.linspace(0,1,8)
y_exact8 = Exact(x8)
y_exact_D_8 = Exact_deri(8)
result8 = np.sqrt(mean_squared_error(y_exact8,y8))
error8 = max(abs(y_exact8 - y8))
Error.append(error8)
RMSE.append(result8)
List8 = {"x":x8,"y_num":y8,"y_exact":y_exact8}
num8 = pd.DataFrame(List8)
print(num8)

print("####################### N = 16 ############################")
y16,y_16 =Lin_shooting_RD(0,1,func,0,1,0.25,0,16,0.0000001)
x16 = np.linspace(0,1,16)
y_exact16 = Exact(x16)
y_exact_D_16 = Exact_deri(16)
result16 = np.sqrt(mean_squared_error(y_exact16,y16))
error16 = max(abs(y_exact16 - y16))
Error.append(error16)
RMSE.append(result16)
List16 = {"x":x16,"y_num":y16,"y_exact":y_exact16}
num16 = pd.DataFrame(List16)
print(num16)

print("####################### N = 32 ############################")
y32,y_32 = Lin_shooting_RD(0,1,func,0,1,0.25,0,32,0.0000001)
x32 = np.linspace(0,1,32)
y_exact32 = Exact(x32)
y_exact_D_32 = Exact_deri(32)
result32 = np.sqrt(mean_squared_error(y_exact32,y32))
error32 = max(abs(y_exact32 - y32))
Error.append(error32)
RMSE.append(result32)
List32 = {"x":x32,"y_num":y32,"y_exact":y_exact32}
num32 = pd.DataFrame(List32)
print(num32)

print("####################### N = 64 ############################")
y64,y_64 = Lin_shooting_RD(0,1,func,0,1,0.25,0,64,0.0000001)
x64 = np.linspace(0,1,64)
y_exact64 = Exact(x64)
y_exact_D_64 = Exact_deri(64)
result64 = np.sqrt(mean_squared_error(y_exact64,y64))
error64 = max(abs(y_exact64 - y64))
Error.append(error64)
RMSE.append(result64)
List64 = {"x":x64,"y_num":y64,"y_exact":y_exact64}
num64 = pd.DataFrame(List64)
print(num64)

plt.figure(figsize = (15,10))
display(plt.plot(x2,y2,label = "y_num,N = 2"))
display(plt.scatter(x2,y_exact2,label = "y_exact,N = 2"))

display(plt.plot(x4,y4,label = "y_num,N = 4"))
display(plt.scatter(x4,y_exact4,label = "y_exact,N = 4"))

display(plt.plot(x8,y8,label = "y_num,N = 8"))
display(plt.scatter(x8,y_exact8,label = "y_exact,N = 8"))

display(plt.plot(x16,y16,label = "y_num,N = 16"))
display(plt.scatter(x16,y_exact16,label = "y_exact,N = 16"))

display(plt.plot(x32,y32,label = "y_num,N = 32"))
display(plt.scatter(x32,y_exact32,label = "y_exact,N = 32"))

display(plt.plot(x64,y64,label = "y_num,N = 64"))
display(plt.scatter(x64,y_exact64,label = "y_exact,N = 64"))
display(plt.title("plot Between x and y "))
display(plt.xlabel("x"))
display(plt.ylabel("y"))
display(plt.grid())
display(plt.legend())
display(plt.show())

plt.figure(figsize = (15,10))
display(plt.plot(x2,y_2,label = "y_num,N = 2"))
display(plt.scatter(x2,y_exact_D_2,label = "y_exact,N = 2"))

display(plt.plot(x4,y_4,label = "y_num,N = 4"))
display(plt.scatter(x4,y_exact_D_4,label = "y_exact,N = 4"))

display(plt.plot(x8,y_8,label = "y_num,N = 8"))
display(plt.scatter(x8,y_exact_D_8,label = "y_exact,N = 8"))

display(plt.plot(x16,y_16,label = "y_num,N = 16"))
display(plt.scatter(x16,y_exact_D_16,label = "y_exact,N = 16"))

display(plt.plot(x32,y_32,label = "y_num,N = 32"))
display(plt.scatter(x32,y_exact_D_32,label = "y_exact,N = 32"))

display(plt.plot(x64,y_64,label = "y_num,N = 64"))
display(plt.scatter(x64,y_exact_D_64,label = "y_exact,N = 64"))
display(plt.title(" plot Between x and dy/dx "))
display(plt.xlabel("x"))
display(plt.ylabel("dy/dx"))
display(plt.title(""))
display(plt.grid())
display(plt.legend())
display(plt.show())


List = {"Error":Error,"RMSE":RMSE}
num = pd.DataFrame(List)
print(num)


# In[112]:


N = [2,4,8,16,32,64]

x= np.array(N).reshape(-1,1)
y=np.array(Error)
model = LinearRegression().fit(x,y)
r_sq = model.score(x,y)
print('intercept:  ' ,model.intercept_)
print('slope :  ' ,model.coef_)

plt.scatter(x,y)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("log(N)")
plt.ylabel("log(E_max)")
plt.title("log(N) vs log(E_max)")
plt.grid()
plt.show()

