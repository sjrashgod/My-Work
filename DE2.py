import numpy as np
import matplotlib.pyplot as plt

def q1(x):
    k = 2 ; m = 10 ; b = 0.2
    derivative = np.zeros(2)
    derivative[0] = x[1]
    derivative[1] = (-b/m)*x[1] -k/m * x[0]
    return derivative

def rk2_1(x_o,v_o,h,N,t):
    x = x_o ; v = v_o 
    dis = [x] ; vel = [v]
    for i in range(0,N):
        X = [x,v]
        k1 = h * q1(X)
        X = [x + k1[0],v + k1[1]]
        k2 = h * q1(X)

        x = x + 1/2 * (k1[0] + k2[0])
        v = v + 1/2 * (k1[1] + k2[1])
        dis.append(x) ; vel.append(v)
    plt.plot(t,dis)
    plt.plot(t,vel)
    plt.show()

def rk4_1(x_o,v_o,h,N,t):
    x = x_o ; v = v_o 
    dis = [x] ; vel = [v] 
    for i in range(0,N):
        X = [x,v]
        k1 = h * q1(X)
        X = [x + k1[0]/2,v + k1[1]/2]
        k2 = h * q1(X)
        X = [x + k2[0]/2,v + k2[1]/2]
        k3 = h * q1(X)
        X = [x + k3[0],v + k3[1]]
        k4 = h * q1(X)
        
        x = x + 1/6 * (k1[0] + 2*(k2[0] + k3[0]) + k4[0])
        v = v + 1/6 * (k1[1] + 2*(k2[1] + k3[1]) + k4[1])
        dis.append(x) ; vel.append(v)
    plt.plot(t,dis)
    plt.plot(t,vel)
    plt.show()

if __name__ == "__main__":
    # INITIAL CONDITIONS
    # Ist part
    initial_x1 = 2 ; initial_v1 = 0
    N1 = 1000 ; a1 = 0 ; b1 = 100
    h1 = (b1-a1)/N1
    t1 = np.linspace(a1,b1,N1+1)

    rk21 = rk2_1(initial_x1,initial_v1,h1,N1,t1)
    rk41 = rk4_1(initial_x1,initial_v1,h1,N1,t1)
