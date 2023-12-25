# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 21:08:05 2022

@author: harsh
"""

import numpy as np
from Myintegration import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import legendre,eval_legendre
import sympy as sym

plt.style.use("bmh")



def inner_prod(f1,f2,a,b,n):
    new_f = lambda x: f1(x)*f2(x)
    prod = MyLegQuadrature(new_f,a,b,n,100)
    return prod

def leg_fourier(f,n):
    Coeff = np.zeros(n)
    for i in range(n):
        Coeff[i] = (inner_prod(f,legendre(i),-1,1,7))/(inner_prod(legendre(i),legendre(i),-1,1,7))
    return Coeff    


def func1(x):
    return 2 + 3*x + 2*x**4

def func2(x):
    return np.cos(x)*np.sin(x)

jnj = leg_fourier(func1, 5)
print(jnj)

khh = leg_fourier(func2, 10)
print(khh)

def partial_series(f,x,n):
    coeff = leg_fourier(f,n)
    n_arr = np.arange(0,len(coeff))
    return eval_legendre(n_arr,x).dot(coeff)

partial_series = np.vectorize(partial_series)

def Compare_original(f,n_max,d):
    x = np.linspace(-1,1)
    for i in range(1,n_max):
        anay = f(x)
        new = partial_series(f,x,i)
        if max(abs((new - anay)/anay)) <= 0.5/10**d:
           return new,i 

#print(Compare_original(lambda x: np.cos(x)*np.sin(x),100,8))            

x_new = np.linspace(-2,2,50)
n1 = 6
n2 = 10
fig,(ax1,ax2)  = plt.subplots(1,2)
for i in range(1,n1):
    z = partial_series(func1,x_new,i)
    ax1.plot(x_new,z,label = f'for n = {i} terms')
    ax1.legend()
ax1.set_xlabel('value of x')
ax1.set_ylabel('Series calculated')
ax1.set_title('Series calculated for polynomial $2 + 3x + 2x^{4}$')
for j in range(2,12,2):
    m = partial_series(func2,x_new,j)
    ax2.plot(x_new,m,label = f'for n = {j} terms')
    ax2.legend()
ax2.set_xlabel('value of x')
ax2.set_ylabel('Series calculated')
ax2.set_title('Series calculated for function $cos(x)sin(x)$')


