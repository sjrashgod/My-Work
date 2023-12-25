import numpy as np
import math
from scipy.special import legendre
import matplotlib.pyplot as plt
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


def P(n,x):                  #Legendre
    e=[]
    if n%2 == 0:
        m=int(n/2)
    else :
        m = int((n-1)/2)   
    s = np.arange(0,m+1,1)
    for i in s:
        polyn = (((-1)**i*(math.factorial(2*n-2*i)))*(x**(n-2*i)))/((2**n)*(math.factorial(i))*(math.factorial(n-i))*(math.factorial(n-2*i)))
        e.append(polyn)
    return sum(e)



def P_der(n,x):               #Derivative
    e=[]
    if n%2 == 0:
        m=int(n/2)
    else :
        m = int((n-1)/2)
    s = np.arange(0,m+1,dtype=float)
    for i in s:
        polyn= (((-1)**i*gamma_(2*n-2*i+1)*(n-2*i))*( x**(n-2*i-1)))/(2**n*gamma_(i+1)*gamma_(n-i+1)*gamma_(n-2*i+1))
        e.append(polyn)
    return sum(e)
def verify(n,x):              #Verificatiion(inbuilt functions)
    Pn = legendre(n)
    y = Pn(x)
    Pn_d= Pn.deriv()  
    evald = np.polyval(Pn_d,x)  
    return y,evald

"=========LEGENDRE FUNCTION VALUE AFTER SUM OF EACH TERM OF THE SERIES EXPANSION:================"
n=int(input("ENTER THE VALUE of n:"))
x=int(input("ENTER THE VALUE OF x:"))
print("LEGENDRE FUNCTION VALUE AFTER SUM OF EACH TERM OF THE SERIES EXPANSION:")
print(P(n,x))
print("DERIVATIVE OF LEGENDRE FUNCTION VALUE AFTER SUM OF EACH TERM OF THE SERIES EXPANSION:")
print(P_der(n,x))
print("ACTUAL VALUE OF LEGENDRE FUNCTION AND ITS DERIVATIVE")
print(verify(n,x))

#2(a)
p0=[];p1=[];p2=[];p3=[]
x=np.linspace(-1,1,100)
for i in x:
    p0.append(P(0,i))
    p1.append(P(1,i))
    p2.append(P(2,i))
    p3.append(P(3,i))
DataOut = np.column_stack((x,p0,p1,p2,p3))
Data1={"x":x,"P0(x)":p0,"P1(x)":p1,"P2(x)":p2,"P3(x)":p3}
print("VALUES OF LEGENDRE POLYNOMIAL")
print(pd.DataFrame(Data1))
np.savetxt("E:\\SEM-3\\PYTHON SEM 3\\2nd ASSISGNMENT\\leg00.dat", DataOut,delimiter=',')
v=np.loadtxt("E:\\SEM-3\\PYTHON SEM 3\\2nd ASSISGNMENT\\leg00.dat",unpack=True,delimiter=',',dtype='float')
#print(v)
for u in range(1,5):
    plt.plot(v[0],v[u])
plt.xlabel("x")
plt.ylabel("Pn")
plt.grid()
plt.title("Pn vs x ")
plt.savefig("ASSIGNMENT 2(A).PNG")
plt.show()


d_p1=[];d_p2=[];d_p3=[]                # 2(b)
for i in x:
    d_p1.append(P_der(1,i))
    d_p2.append(P_der(2,i))
    d_p3.append(P_der(3,i))
DataOut1 = np.column_stack((x,d_p1,d_p2,d_p3))
Data2={"x":x,"P'1(x)":d_p1,"P'2(x)":d_p2,"P'3(x)":d_p3}
print("VALUES OF DERIVATIVE OF LEGENDRE POLYNOMIAL")
print(pd.DataFrame(Data2))
np.savetxt("E:\\SEM-3\\PYTHON SEM 3\\2nd ASSISGNMENT\\leg01.dat", DataOut1,delimiter=',')
r=np.loadtxt("E:\\SEM-3\\PYTHON SEM 3\\2nd ASSISGNMENT\\leg01.dat",unpack=True,delimiter=',',dtype='float')


plt.plot(r[0],d_p1,label= "P'1")
plt.plot(r[0],d_p2,label= "P'2")
plt.plot(r[0],d_p3,label="P'3 ")
plt.xlabel("x")
plt.ylabel("P'n")
plt.grid()
plt.legend()
plt.title("P'n vs x")
plt.savefig("ASSIGNMENT 2(B).PNG")
plt.show()

#2(c)
def rec():
    rhs1=[]
    lhs1=[]
    for i in x:
        rhs1.append(2*P(2,i))
        lhs1.append(i*P_der(2,i) - P_der(1,i))
    dict1={"x":x,"n":n,"n-1":n-1,"P2(x)":rhs1,"x*P'2(x)-P'1(x)":lhs1}
    print("VERIFYING THE RECURRENCE RELATION 1")
    print(pd.DataFrame(dict1))
    DataOut2 = np.column_stack((x,rhs1,lhs1))
    np.savetxt("E:\\SEM-3\\PYTHON SEM 3\\2nd ASSISGNMENT\\leg02.dat", DataOut2,delimiter=',')

    rhs2 = []
    lhs2 = []
    for i in x:
        rhs2.append(i*P(2,i))
        lhs2.append(3*P(3,i) - 2*P(1,i))
    dict1={"x":x,"n":n,"n-1":n-1,"5*x*P2(x)":rhs2,"3*(P3)+2*(P1)":lhs2}
    print("VERIFYING THE RECURRENCE RELATION 2")
    print(pd.DataFrame(dict1))
    DataOut3 = np.column_stack((x,rhs2,lhs2))
    np.savetxt("E:\\SEM-3\\PYTHON SEM 3\\2nd ASSISGNMENT\\leg03.dat", DataOut3,delimiter=',')

    rhs3 = []
    lhs3 = []
    for i in x:
        rhs3.append(2 * P(2, i))
        lhs3.append(3 * i * P(1, i) - 1 * P(0, i))
    dict1 = {"x": x, "n": n, "n-1": n - 1, "2*P2(x)": rhs3, "(3*x*P1)-(1)*P0(x)": lhs3}
    print("VERIFYING THE RECURRENCE RELATION 3")
    print(pd.DataFrame(dict1))
    DataOut3 = np.column_stack((x, rhs3, lhs3))
    np.savetxt("E:\\SEM-3\\PYTHON SEM 3\\2nd ASSISGNMENT\\leg03.dat", DataOut3, delimiter=',')

rec()