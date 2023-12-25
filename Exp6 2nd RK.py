import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def q1(x,b,n):
    if n == 1:
        derivative = np.zeros(2)
        derivative[0] = x[1]
        derivative[1] = (-b/m)*x[1] -k/m * x[0]
        return derivative
    elif n == 2:
        derivative = np.zeros(2)
        derivative[0] = x[1]
        derivative[1] = (-g/l)*x[0]
        return derivative

def rk2_1(x_o,v_o,h,N,b,n):
    x = x_o ; v = v_o ; all1 = []
    dis = [x] ; vel = [v]
    for i in range(0,N):
        X = [x,v]
        k1 = h * q1(X,b,n)
        X = [x + k1[0],v + k1[1]]
        k2 = h * q1(X,b,n)

        x = x + 1/2 * (k1[0] + k2[0])
        v = v + 1/2 * (k1[1] + k2[1])
        dis.append(x) ; vel.append(v)
    all1.extend([dis,vel])
    return all1

def graph(rk21,rk2_2u,rk2_2c,rk2_2o,rk2_3,t1_a,t1_b,t2):
    fig,ax =  plt.subplots()
    fig1,ax1 = plt.subplots(1,2)
    fig2,ax2 = plt.subplots()
    ax.plot(t1_a,rk21[0],label = "Displacement")
    ax.plot(t1_a,rk21[1],label = "Velocity")
    ax1[0].plot(t1_b,rk2_2u[0],label = "Underdamped")
    ax1[0].plot(t1_b,rk2_2c[0],label = "Critical Damped")
    ax1[0].plot(t1_b,rk2_2o[0],label = "Overdamped")
    ax1[1].plot(t1_b,rk2_2u[1],label = "Underdamped")
    ax1[1].plot(t1_b,rk2_2c[1],label = "Critical Damped")
    ax1[1].plot(t1_b,rk2_2o[1],label = "Overdamped")
    ax2.plot(t2,rk2_3[0],label = "Angular Displacement")
    ax2.plot(t2,rk2_3[1],label = "Angular Velocity")
    ax.set(xlabel = "Time / Time period",ylabel = "Displacement (cm)/Velocity (cm/sec)",title = "SIMPLE HARMONIC OSCILLATOR")
    ax1[0].set(xlabel = "Time / Time period",ylabel = "Displacement (cm)",title = "DAMPED HARMONIC OSCILLATOR")
    ax1[1].set(xlabel = "Time / Time period",ylabel = "Velocity (cm/sec)",title = "DAMPED HARMONIC OSCILLATOR")
    ax2.set(xlabel = "Time / Time period",ylabel = "Angular displacement (radian)/Angular velocity (radian/sec)",title = "SIMPLE PENDULUM")
    ax.grid(ls = "--")
    ax1[0].grid(ls = "--")
    ax1[1].grid(ls = "--")
    ax2.grid(ls = "--")
    ax.legend()
    ax1[0].legend()
    ax1[1].legend()
    ax2.legend(loc = "lower right")
    plt.show()


def table(rk21,rk2_2u,rk2_2c,rk2_2o,rk2_3,t1_a,t1_b,t2):
    table1 = PrettyTable(['Sr No.','Time/Time period','Displacement (cm)','Velocity (cm/sec)'])
    table2 = PrettyTable(['Sr No.','Time/Time period','Displacement (cm) (Underdamped)','Velocity (cm/sec) (Underdamped)'])
    table3 = PrettyTable(['Sr No.','Time/Time period','Displacement (cm) (Critical Damped)','Velocity (cm/sec) (Critical Damped)'])
    table4 = PrettyTable(['Sr No.','Time/Time period','Displacement (Overdamped)','Velocity (cm/sec) (Overdamped)'])
    table5 = PrettyTable(['Sr No.','Time/Time period','Theta (degree)','Omega (degree/sec)'])
    for i in range(len(t1_a)):
        table1.add_row([i+1,t1_a[i],rk21[0][i],rk21[1][i]])
        table5.add_row([i+1,t2[i],rk2_3[0][i],rk2_3[1][i]])
    for i in range(len(t1_b)):
        table2.add_row([i+1,t1_b[i],rk2_2u[0][i],rk2_2u[1][i]])
        table3.add_row([i+1,t1_b[i],rk2_2c[0][i],rk2_2c[1][i]])
        table4.add_row([i+1,t1_b[i],rk2_2o[0][i],rk2_2o[1][i]])
    print("\nSimple Harmonic Oscillator (RK2):\n",table1)
    print("\nDamped Harmonic Oscillator (Underdamped) (RK2):\n",table2)
    print("\nDamped Harmonic Oscillator (Critical Damped) (RK2):\n",table3)
    print("\nDamped Harmonic Oscillator (Overdamped) (RK2):\n",table4)
    print("\nSimple Pendulum (RK2):\n",table5)

if __name__ == "__main__":
    # INITIAL CONDITIONS

    # Ist PART & IInd PART
    k = 0.5 ; m = 1 ; t_period = []
    initial_x1 = 2 ; initial_v1 = 0
    b = [0,0.1,np.sqrt(2),2.0] # DAMPING COFFICIENT
    for i in range(2):
        T1 = 2*np.pi/(np.sqrt(k/m -(b[i]**2/(m**2)*4))) # TIME PERIOD
        t_period.append(T1)
    a1_a = 0 ; b1_a = 5*t_period[0] # SIMPLE HARMONIC OSCILLATOR
    h1_a = t_period[0]/100 ; N1_a = int((b1_a-a1_a)/h1_a)
    t1_a = np.linspace(a1_a,b1_a,N1_a+1)/t_period[0]

    a1_b = 0 ; b1_b = 5*t_period[1] # DAMPED HARMONIC OSCILLATOR
    h1_b = t_period[1]/100 ; N1_b = int((b1_b-a1_b)/h1_b)
    t1_b = np.linspace(a1_b,b1_b,N1_b+1)/t_period[1]

    # IIIrd PART
    g = 9.8 ; l = 1
    initial_theta = 2 ; initial_omega = 0
    T2 = 2*np.pi*np.sqrt(l/g)
    a2 = 0 ; b2 = 5*T2
    h2 = T2/100 ; N2 = int((b2-a2)/h2)
    t2 = np.linspace(a2,b2,N2+1)/T2

    rk21 = rk2_1(initial_x1,initial_v1,h1_a,N1_a,b[0],1) # SOLUTION OF FIRST PART

    rk2_2u = rk2_1(initial_x1,initial_v1,h1_b,N1_b,b[1],1) # SOLUTION OF SECOND PART
    rk2_2c = rk2_1(initial_x1,initial_v1,h1_b,N1_b,b[2],1)
    rk2_2o = rk2_1(initial_x1,initial_v1,h1_b,N1_b,b[3],1) 

    rk2_3 = rk2_1(initial_theta,initial_omega,h2,N2,0,2) # SOLUTION OF THIRD PART

    table(rk21,rk2_2u,rk2_2c,rk2_2o,rk2_3,t1_a,t1_b,t2)
    graph(rk21,rk2_2u,rk2_2c,rk2_2o,rk2_3,t1_a,t1_b,t2)
