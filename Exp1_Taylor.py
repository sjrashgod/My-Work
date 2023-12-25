import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def MySinSeries(x,n):
    list=[]
    for y in x:
        sin=0
        for i in range(n):
             sin+=(-1)**i*y**((2*i)+1)/math.factorial((2*i)+1)
        list.append(sin)
    return list

def MyCosSeries(x,n):
    list=[]
    for y in x:
        cos=0
        for i in range(n):
            cos+=(-1)**i*y**(2*i)/math.factorial(2*i)      
        list.append(cos)
    return list


#Q1(a)      
x=np.linspace(-2*np.pi,2*np.pi,200)
m=[1,2,5,10,20] 
sin_list=[]
cos_list=[]
for n in m:
    sin_list.append(MySinSeries(x,n))
    cos_list.append(MyCosSeries(x,n))

#Q1(a) Creating sin and cos inbuilt arrays for comparison  
sin_in=np.sin(x)
cos_in=np.cos(x)  

#Q1(b) & (c) for both sin(x) and cos(x)    
x0=[np.pi/4]
s=np.arange(2,22,2)
y0_sin=[]
y0_cos=[]
for n in s:
    y0_sin.append(MySinSeries(x0,n))
    y0_cos.append(MyCosSeries(x0,n))

#Q1(b) & (c) Creating sin and cos inbuilt arrays for comparison  
sin_inbuilt=np.array([np.sin(x0)]*len(s))
cos_inbuilt=np.array([np.cos(x0)]*len(s))    

plt.plot(x,sin_list[0], label = "m = 1")
plt.plot(x,sin_list[1], label = "m = 2")
plt.plot(x,sin_list[2], label = "m = 5")
plt.plot(x,sin_list[3], label = "m = 10")
plt.plot(x,sin_list[4], label = "m = 20")
plt.plot(x,sin_in, label="inbulit")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("Q1(a) Comparison of MySinX with inbuilt sin(x)")
plt.legend(loc='upper right')
plt.grid()
plt.show()

plt.plot(x,cos_list[0], label = "m = 1")
plt.plot(x,cos_list[1], label = "m = 2")
plt.plot(x,cos_list[2], label = "m = 5")
plt.plot(x,cos_list[3], label = "m = 10")
plt.plot(x,cos_list[4], label = "m = 20")
plt.plot(x,cos_in, label="inbulit")
plt.xlabel("x")
plt.ylabel("cos(x)")
plt.title("Q1(a) Comparison of MyCosX with inbuilt cos(x)")
plt.legend(loc='upper right')
plt.grid()
plt.show()

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(s,sin_inbuilt,label="Inbuilt for $x_0 = \pi/4$")
ax1.plot(s,y0_sin,marker="*", label="MySin$(x_0)$ for $x_0 = \pi/4$")
ax1.set(xlabel="n",ylabel="sin(x)")
ax1.legend()
ax1.grid()
ax2.plot(s,cos_inbuilt,label="Inbuilt for $x_0 = \pi/4$")
ax2.plot(s,y0_cos,marker="*", label="MyCos$(x_0)$ for $x_0 = \pi/4$")
ax2.set(xlabel="n",ylabel="cos(x)")
ax2.legend()
ax2.grid()
fig.suptitle("Q1(c) Comparison of MySin$(x_0)$ with inbuilt $sin(x_0)$",)
plt.show()

def mfun(x,tol):
    e=0
    n=0
    lis=[]
    while True:
        e=e+(-1)**n*x**((2*n)+1)/math.factorial((2*n)+1)
        lis.append(e)
        n+=1
        if len(lis)>=2:
            err = abs((lis[-1]-lis[-2])/lis[-2])
            if err <= tol:
                break
    return e,n

def mysin(x_a):
    sigd=float(input("Enter the no of significant digits :"))
    tol = 10**-(sigd)
    r_a=[];n_a=[]
    for x in x_a:
        h=mfun(x,tol)
        r_a.append(h[0]);n_a.append(h[1])
    return r_a,n_a

x=np.linspace(1,np.pi,8)
sin_in=np.sin(x)
d=mysin(x)
data={"x":x,"sin(x)_calc":d[0],"n":d[1],"sin(x)_inbuilt":sin_in}
print(pd.DataFrame(data))
x=np.linspace(1,np.pi,8)
sin_in=np.sin(x)
plt.plot(x,d[0],label="calculated",marker="o")
plt.plot(x,sin_in,label="inbuilt")
plt.xlabel("x")
plt.ylabel("sin x")
plt.title("Q2 Graph for sin(x) using tolerance")
plt.legend()
plt.grid()
plt.show()
