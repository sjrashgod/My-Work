import numpy as np
import math
import matplotlib.pyplot as plt

def trapz(f,a,b,n):
    h=(b-a)/n
    y=[]
    for i in range(n+1):    
        y.append(f(a+i*h))  #Value of f(x) at the nodal points
    trap=h*(f(a)+f(b))/2
    for j in range(1,len(y)-1):
        trap=trap+h*(y[j])
    return(trap)

def simps(f,a,b,n):
    h2=(b-a)/(2*n)
    simp=h2*(f(a)+f(b))/3
    for i in range(1,2*n): 
        if(i%2==0):
            simp=simp+2*h2*f(a+i*h2)/3  #y at Even Nodes 
        elif(i%2==1):
            simp=simp+4*h2*f(a+i*h2)/3  #y at Odd Nodes
    return(simp)

def grf(f,a,b):
    N=np.arange(1,100,1)    #Number of Subintervals
    H=(b-a)/N
    It=[]
    for i in N:
        z=trapz(f,a,b,i)
        It.append(z)
    plt.yscale("log")
    plt.xscale("log")
    plt.plot(H,It,label="Trapezoidal",marker=".")
    
    H2=(b-a)/(2*N)
    Is=[]
    for j in 2*N:
        z=simps(f,a,b,j)
        Is.append(z)
    plt.plot(H2,Is,label="Composite Simpson",marker=".")
    plt.legend()
    plt.xlabel("h")
    plt.ylabel("I(h)")
    plt.title("Convergence Test - Samarth Jain")
    plt.show()

def Q2a():
    #Trapezoidal
    f=eval("lambda x:"+input("Enter the Function : "))
    a=float(input("a = "))
    b=float(input("b = "))
    n=int(input("N = "))
    trap=trapz(f,a,b,n)
    print("Integral is {:.8} using Trapezoidal Method".format(float(trap)))
    
def Q2b():
    #Simpson
    f=eval("lambda x:"+input("Enter the Function : "))
    a=float(input("a = "))
    b=float(input("b = "))
    n=int(input("N = "))
    simp=simps(f,a,b,n)
    print("Integral is {:.8} using Simpson Method".format(float(simp)))

def Q2c():
    #Example function = x**2
    #Analytical Integral = x**3/3
    f=eval("lambda x:"+"x**2")
    a=float(input("a = "))
    b=float(input("b = "))
    n=int(input("N = "))
    anlytc=(b**3-a**3)/3
    trap=trapz(f,a,b,n)
    err=abs(anlytc-trap)
    print("Truncation Error = {:e}".format(float(err)))

def Q2d():
    f=eval("lambda x:"+input("Enter the Function : "))
    a=float(input("a = "))
    b=float(input("b = "))
    grf(f,a,b)


def Q3a():
    v=np.array([0.0, 0.5, 2.0, 4.05, 8.0, 12.5, 18.0, 24.5, 32.0, 40.5, 50.0])
    c=np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    h=(c[-1]-c[0])/(len(c)-1)
    print(h)
    pwr=h*(v[0]+v[-1])/2
    for j in range(1,len(v)-1):
        pwr=pwr+h*(v[j])
    print("Power is {:.8} Joules".format(float(pwr)))

def Q3b():
    f=eval("lambda x:"+input("Enter the Function : "))
    a=float(input("a = "))
    b=float(input("b = "))
    n=int(input("N = "))
    h=(b-a)/n
    #Trapezoidal
    trap=trapz(f,a,b,n)
    print("Integral is {:.8} using Trapezoidal Method".format(float(trap)))
    #Simpson
    simp=simps(f,a,b,n)
    print("Integral is {:.8} using Simpson Method".format(float(simp)))
    
    print("f(h) = {:.8}".format(float(f(h))))
    
    #Plotting
    grf(f,a,b)
