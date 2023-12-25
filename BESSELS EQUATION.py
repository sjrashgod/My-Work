import numpy as np
import math
import pandas as pd

def gamma_(a):                           #Gamma Function
    if a==0.5:
        return np.sqrt(np.pi)
    elif a==1:
        return(a)
    else:
        return (a-1)*gamma_(a-1)
G1=float(input("ENTER NUMBER: "))
print("GAMMA OF THIS NUMBER: ",gamma_(G1))

n=int(input("Enter The value Of n: "))
x=int(input("Enter The Value Of x: "))

def Jn(n,x):
    r=[]
    L=[]
    for i in range(50):
        Jn= ((-1)**i/((math.factorial(i))*(gamma_(n+i+1)))) * ((x/2)**(n+2*i))
        L.append(Jn)
    return sum(L)

print(Jn(n,x))

p0=[]; p1=[]; p2=[]
x=np.linspace(-1,1,100)
for i in x:
    p0.append(Jn(0,i))
    p1.append(Jn(1,i))
    p2.append(Jn(2,i))

Data1={"x":x,"J0(x)":p0,"J1(x)":p1,"J2(x)":p2}
print("VALUES OF LEGENDRE POLYNOMIAL")
print(pd.DataFrame(Data1))