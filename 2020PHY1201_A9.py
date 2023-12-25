import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from MyIVP1 import rk4
from scipy import stats

# Name => Sarthak Jain   Roll no. => 2020PHY1201
# Patner's Name => Ishmeet Singh   Roll no. => 2020PHY1221

def Func(x,var):
    y,z=var
    f1=z
    f2=np.sin(3*x) - y
    t=np.array([f1,f2])
    return t 

def secant(x0,x1,P0,P1):
    x2 = x1 - ((x1-x0)/(P1-P0))*P1
    return x2

def Lin_shooting(Func,a,b,N,x0,x1,tol,nmax):

    IC_1=np.array([-1-x0,x0],dtype=float)
    s1=RK4(Func,IC_1,a,b,N)
    y1,y1_der=s1[0].T

    IC_2=np.array([-1-x1,x1],dtype=float)
    s2=RK4(Func,IC_2,a,b,N)
    y2,y2_der=s2[0].T

    yb_der=1           #given boundary condition for y'(pi/2)=1
    
    P0=1-y1_der[-1]    #comparing the boundary values with the values generated by RK4 for initial guess condition.
    P1=1-y2_der[-1]
    l1=[];l2=[];l3=[]
    for i in range(nmax):
        if abs(P0) < tol:
            break
        else:
            new_guess=secant(x0,x1,P0,P1)

            x0=x1
            x1=new_guess
            IC_3=np.array([-1-x1,x1],dtype=float)
            
            z=RK4(Func,IC_3,a,b,N)
            l1,l2=z[0].T
            l3=z[1]

            y_der_new=yb_der - l2[-1]
            P0=P1
            P1=y_der_new

    return l1,l2,l3

def analytic(x):
    z=((3 * np.sin(x)) / 8 - np.cos(x) - (1 / 8) * np.sin(3 * x))
    return z 

def table(Func,a,b,N,x0,x1,tol,nmax):
    y,y_der,x = Lin_shooting(Func,0,np.pi/2,N,x0, x1, tol, nmax)
    print()
    print("for n=",N,"\n")
    lis1 = []
    for i in x:
        lis1.append(analytic(i))
    lis2 = []
    for i in range(len(x)):
        lis2.append(abs(y[i] - lis1[i]))
    data1 = {"x": x, "y": y, "y'": y_der, "y analytic": lis1, "error": lis2}
    print(pd.DataFrame(data1))

x0=1
x1=0
tol=10**(-10)
nmax=100

table(Func,0,np.pi/2,4,x0,x1,tol,nmax)
table(Func,0,np.pi/2,8,x0,x1,tol,nmax)



f1=lambda x: 3/8*np.sin(x)- np.cos(x) - 1/8*np.sin(3*x)
x_t=np.linspace(0,np.pi/2,100)

fig=plt.figure()
(ax1),(ax2) = fig.subplots(2,1)
for i in range(0,9):
    N=2**i
    arr1,arr2,arr3=Lin_shooting(Func,0,np.pi/2,N,x0,x1,tol,100)
    ax1.plot(arr3,arr1,label="y at N="+str(N),ls="dashdot")
    ax2.plot(arr3,arr2,label="y' at N="+str(N),ls="dashdot")

ax1.plot(x_t,f1(x_t),label="analytic")
ax2.set_title("$dy/dx$ vs $x$")
ax2.set_xlabel("$x$")
ax2.set_ylabel("$dy/dx$")
ax2.legend()
ax2.grid()
ax1.set_title("$y$ vs $x$")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.legend()
ax1.grid()
plt.tight_layout()
fig.suptitle("Sarthak Jain and Ishmeet Singh")
plt.show()


def error(a):
    y_arr,y_Der_arr,x_arr=Lin_shooting(Func,0,np.pi/2,a,0,1,tol,nmax)
    y_analytic=f1(x_arr)
    err_rms=0
    err_max=0
    for j in range(len(x_arr)):
        err_rms += (y_arr[j] - y_analytic[j])**2
    err_rms=np.sqrt((err_rms/a))
    err_max=max(abs(y_arr-y_analytic))
    
    return x_arr,y_arr,y_analytic,err_rms,err_max

N_val=[]
for k in range(1,8):
    N=2**k
    N_val.append(N)
print()
    
l11,l22,l33,l44,l55=[],[],[],[],[]
ratio_rms=["none"];ratio_max=["none"]

for i in N_val:
    t1,t2,t3,t4,t5=error(i)
    l11.append(t1),l22.append(t2),l33.append(t3),l44.append(t4),l55.append(t5)
    if i!=2:
        ratio_rms.append(l55[-2]/l55[-1])
        ratio_max.append(l44[-2]/l44[-1])

data4={"n values":N_val,'rms error':l55,'ratio of En/E2n':ratio_rms}
print("\n",pd.DataFrame(data4))

data5={"n values":N_val,'max error':l44,'ratio of En/E2n':ratio_max}
print("\n",pd.DataFrame(data5))


fig=plt.figure()
ax1=fig.subplots(1,1)
ax1.scatter(N_val,l55,label="RMS ERROR")
ax1.plot(N_val,l55,ls="--")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_title("log(N) vs log(E)")
ax1.set_xlabel("log(N)")
ax1.set_ylabel("log(E)")
ax1.legend()
ax1.grid()
fig.suptitle("Sarthak Jain and Ishmeet Singh")
plt.tight_layout()
plt.show()

slope=stats.linregress(np.log(N_val),np.log(l55))
print("\nslope for line log(N) and log(E) is: ",slope[0])
