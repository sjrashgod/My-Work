import numpy as np
import matplotlib.pyplot as plt

def step_size(a,b,N):
    h = (b-a)/N
    return h

def trap_method(h,V,I):
    z2 = 0
    z1 = V[0] + V[10]
    for i in range(1,len(V)-1):
        z2 = z2 + V[i]

    i_trap = h/2 * (z1 + 2*z2)
    return i_trap

def simp_method(h,V,I):
    z2 = 0
    z3 = 0
    z1 = V[0] + V[10]
    if (N%2)==0:
        for i in range(1,len(V)-1):
            if i % 2 == 0:
                z2 = z2 + V[i]
            else:
                z3 = z3 + V[i]
        i_simp = h/3 * (z1 + 4 * z3 + 2 * z2)
    else:
        print("simpson method not applicable")
    return i_simp

def plot(V, I):
    plt.scatter(I,V)
    plt.show()

if __name__ == "__main__":
        V = np.array([0.0, 0.5, 2.0, 4.05, 8.0, 12.5, 18.0, 24.5, 32.0, 40.5, 50.0])
        I = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        a = I[0]
        b = I[10]
        N = len(I)-1
        h = step_size(a,b,N)
        i_trap = trap_method(h,V,I)
        i_simp = simp_method(h,V,I)
        plot(V, I)
        print(h)
        print(i_trap)
        print(i_simp)

