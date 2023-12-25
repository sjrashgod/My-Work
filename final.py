#Name=Pawanpreet Kaur
#College Roll No. = 2020PHY1092
#University Roll no. = 20068567038

from MyIntegration import MySimp
from MyIntegration import MyTrap
from MyIntegration import MyLegQuadrature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate
from legendre import Lege
from scipy.special import legendre
from sympy import *
from sympy import simplify

#(a)
#coefficient calculation
def coeff(f,n,n0=4,m=1):
  x=symbols('x')
  f2=Lege(n)
  def f1(*args):
    return f(*args) * f2(*args)
  inte=((2*n+1)/2)*MyLegQuadrature(f1,-1,1,n0,m,key=False,tol=None,m_max=None)
  return inte

#Expansion
def expansion(f,n):
    lis=[]
    x=symbols('x')
    for i in range(0,n):
        f2=Lege(i)
        def s1(*args):
            return coeff(f,i,n0=10,m=100) * f2(*args)
        lis.append(s1(x))
    k=sum(lis)
    p_x=simplify(k)
    fx=lambdify(x,p_x,"math")
    return fx

#(b)        
f1=lambda x : 2 + 3*x + 2*x**4  
f2=lambda x : np.sin(x)*np.cos(x)
t1=[];t2=[];t3=[]
for i in range(0,5):
     
     g=coeff(f1,i,n0=100,m=10000)
     t3.append(g)
     if g<5.97001796356409e-15:
        pass
     else:
        t1.append(g)    
          
for i in range(0,10):
     g=coeff(f2,i,n0=100,m=10000)
     t2.append(g)
         
n=1;tol=0.1e-6
tes=np.linspace(-np.pi,np.pi,100)
old=[]
s=expansion(f2,n)
for x in tes:
    old.append(s(x))
new=[]

while n<100:
      n=n+1
      u=expansion(f2,n)
      for x in tes:
          new.append(u(x))
      e=[]
      for a,b in zip(old,new):
           if b< 6e-15:
              err=abs(b-a)
              e.append(err)
           else:
              e.append((b-a)/b)
      minv=min(e)
      if minv<=tol:
         break
      else:
         old=new
         new=[]
n_tol=n
z1=["C0","C1","C2","C3","C4"]
z2=["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
data1={"Coefficient corresponding to nth Legendre Polynomial":z1,"Value of Coefficient":t3}
print(pd.DataFrame(data1))
print()
print("Non-zero coefficients in the expansion of the function of f (x) = 2 + 3x + 2x**4 :",t1)
print()
data2={"Coefficient corresponding to nth Legendre Polynomial":z2,"Value of Coefficient":t2}
print(pd.DataFrame(data2))
print("Number of terms required in the expansion of sin(x)*cos(x) which result in accuracy of 6 significant digits = ",n_tol)

#(c)
x_a=np.linspace(-2,2,100)
n_a=[1,2,3,4,5]
d1=[]
for n in n_a :
    d1.append(expansion(f1,n))
e1=[];e2=[];e3=[];e4=[];e5=[];e6=[]
for x in x_a:
    e1.append(d1[0](x))
    e2.append(d1[1](x))
    e3.append(d1[2](x))
    e4.append(d1[3](x))
    e5.append(d1[4](x))
    e6.append(f1(x))
    
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Ques 2(c)')
  
ax1.plot(x_a,e1,linestyle='dashed',label="n=1")
ax1.plot(x_a,e2,linestyle='dashed',label="n=2")
ax1.plot(x_a,e3,linestyle='dashed',label="n=3")
ax1.plot(x_a,e4,linestyle='dashed',label="n=4")
ax1.plot(x_a,e5,linestyle='dashed',label="n=5")
ax1.plot(x_a,e6,label="exact")
ax1.grid()
ax1.legend()
ax1.set(xlabel="x",ylabel="f(x)",title="Series calculated for polynomial $2+3x+2x^4$")



n_a=[2,4,6,8,10]
d2=[]
for n in n_a :
    d2.append(expansion(f2,n))
e1=[];e2=[];e3=[];e4=[];e5=[];e6=[]
for x in x_a:
    e1.append(d1[0](x))
    e2.append(d1[1](x))
    e3.append(d1[2](x))
    e4.append(d1[3](x))
    e5.append(d1[4](x))
    e6.append(f2(x))
    
ax2.plot(x_a,e1,linestyle='dashed',label="n=2")
ax2.plot(x_a,e2,linestyle='dashed',label="n=4")
ax2.plot(x_a,e3,linestyle='dashed',label="n=6")
ax2.plot(x_a,e4,linestyle='dashed',label="n=8")
ax2.plot(x_a,e5,linestyle='dashed',label="n=10")
ax2.plot(x_a,e6,label="exact")
ax2.grid()
ax2.legend()
ax2.set(xlabel="x",ylabel="f(x)",title="Series calculated for $sin(x)*cos(x)$")
plt.show()

