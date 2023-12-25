import numpy as np

def funct(x):
    y=0
    y=np.log(x)*x-1
    return y
def dervfunct(x):
    dry=0
    dry=1+np.log(x)
    return dry
def NewtonRaphson(x):
    epsilon=funct(x)/dervfunct(x)
    q=0
    while abs(epsilon)>=0.000001:
        epsilon=funct(x)/dervfunct(x)
        x=x-epsilon
        q+=1
    print("The value of root is:",x)
    print(q)
    
x0=1
NewtonRaphson(x0)    