import numpy as np
import matplotlib.pyplot as plt

def f1(x,y):
    dx = y + x - x**3
    return dx

def f2(x):
    dy = -x
    return dy

def rk4(h,x_o,y_o,N):
    x = x_o ; y = y_o
    x_list = [x_o] ; y_list = [y_o] ; all = []
    for i in range(0,N):
        k1 = h * f1(x,y)
        l1 = h * f2(x)

        k2 = h * f1(x + (h/2*k1),y + (h/2*k1))
        l2 = h * f2(x + (h/2*l1))

        k3 = h * f1(x + (h/2*k2),y + (h/2*k2))
        l3 = h * f2(x + (h/2*l2))

        k4 = h * f1(x + h*k3 ,y + h*k3)
        l4 = h * f2(x + h*l3)

        x = x + 1/6 * (k1 + 2*(k2 + k3) + k4)
        y = y + 1/6 * (l1 + 2*(l2 + l3) + l4)
        x_list.append(x) ; y_list.append(y)
    all.extend([x_list,y_list])
    return all

def graph(sol1,sol2,sol3,sol4,t):
    fig,ax =  plt.subplots(2,2)
    fig1,ax1 = plt.subplots(2,2)
    fig2,ax2 = plt.subplots(2,2)
    ax[0,0].scatter(sol1[0],sol1[1], c = "green")
    ax[0,1].scatter(sol2[0],sol2[1], c = "green")
    ax[1,0].scatter(sol3[0],sol3[1], c = "green")
    ax[1,1].scatter(sol4[0],sol4[1], c = "green")

    ax1[0,0].scatter(t,sol1[0], c = "red")
    ax1[0,1].scatter(t,sol2[0], c = "red")
    ax1[1,0].scatter(t,sol3[0], c = "red")
    ax1[1,1].scatter(t,sol4[0], c = "red")

    ax2[0,0].scatter(t,sol1[1])
    ax2[0,1].scatter(t,sol2[1])
    ax2[1,0].scatter(t,sol3[1])
    ax2[1,1].scatter(t,sol4[1])
    plt.show()

if __name__ == "__main__":
    x_o = 0
    y_o = [-1,-2,-3,-4]
    N = 100
    h = (15 - 0)/N
    t = np.linspace(0,15,N+1)
    sol1 = rk4(h,x_o,y_o[0],N)
    sol2 = rk4(h,x_o,y_o[1],N)
    sol3 = rk4(h,x_o,y_o[2],N)
    sol4 = rk4(h,x_o,y_o[3],N)
    graph(sol1,sol2,sol3,sol4,t)