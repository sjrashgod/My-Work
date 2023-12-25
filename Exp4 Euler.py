import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

"Initial Conditions"
t_half = 4  #Q3A
R = 1000; C = 10 ** -6  #Q3B
m = 50; r = 4 * 10 ** -2; neta = 20  #Q3C
τ = np.array([1 / (0.693 / t_half), R * C, m / (6 * np.pi * neta * r)])
c = [20000, 10, 5] #Initial values for dependent variable

"Function for defining nodal points"        
def nodes(a, b, h):
    N = (b - a) / h
    return N

"Calculations for Step Size"
a = 0 #initial time
b = np.array([5 * τ[0], 5 * τ[1], 5 * τ[2]]) #time till the system needs to be observed
h = np.array([τ[0] / 10, τ[1] / 10, τ[2] / 10]) #step size
N_r = int(nodes(a, b[0], h[0])); N_rc = int(nodes(a, b[1], h[1])); N_st = int(nodes(a, b[2], h[2]))

"Time axes"
t_r = np.linspace(0, b[0], 51)
t_rd = (np.linspace(0, b[0], 51)) / t_half
t_rc = (np.linspace(0, b[1], 51))
t_st = (np.linspace(0, b[2], 51))
"Defining Initial Function"
def func_ini(c, τ):
    f = -(c / τ)
    return f

"Defining Analytic Solution for comparison with Numerical Methods"
def analytic(y_o, t, τ):
    analytic_sol = []
    for i in range(len(t)):
        y = y_o * np.exp(-t[i] / τ)
        analytic_sol.append(y)
    return analytic_sol

"Q2A: Function for Euler Method"
def Euler_Method(h, c, τ, N):
    x = c
    y = [x]
    for i in range(0, N):
        x = x + h * func_ini(x, τ)
        y.append(x)
    return y

"Q2B: Function for Runge Kutta Order 2 Method"
def RK2_Method(h, c, τ, N):
    x = c
    y = [x]
    for i in range(0, N):
        k1 = h * func_ini(x, τ)
        k2 = h * func_ini((x + k1), τ)
        k = (k1 + k2) / 2
        x = x + k
        y.append(x)
    return y

"Function for finding Absolute Error (Analytical - Numerical)"
def Absolute_Error(arr1, arr2):
    error = np.array(arr1) - np.array(arr2)
    return error

"Function for Graphing"
def Graph(t_rd, t_rc, t_st, e1, e2, e3, r1, r2, r3, e_1a, e_1b, e_2a, e_2b, e_3a, e_3b):
    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    ax.scatter(t_rd, e1, c="limegreen", label="Euler Method", s=50)
    ax.scatter(t_rd, r1, c="purple", label="Runge Kutta order 2")
    ax.set(xlabel="Time / $t_{1/2}$", ylabel="No. of Nuclei (N)", title="Numerical Method Graph")
    ax1.scatter(t_rd, e_1a, c="limegreen", label="Absolute Error in Euler Method")
    ax1.scatter(t_rd, e_1b, c="purple", label="Absolute Error in RK2")
    ax1.set(xlabel="Time / $t_{1/2}$", ylabel="Absolute Error", title="Absolute Error Graph")
    ax2.scatter(t_rc, e2, c="limegreen",label="Euler Method", s=50)
    ax2.scatter(t_rc, r2, c="purple", label="Runge Kutta order 2")

    ax2.set(xlabel="Time (seconds)", ylabel="Voltage (V)", title="Numerical Method Graph", xlim=(-0.0005, 0.0055))
    ax3.scatter(t_rc, e_2a, c="limegreen", label="Absolute Error in Euler Method")
    ax3.scatter(t_rc, e_2b, c="purple", label="Absolute Error in RK2")
    ax3.set(xlabel="Time (seconds)", ylabel="Absolute Error", title="Absolute Error Graph", xlim=(-0.0005, 0.0055))
    ax4.scatter(t_st, e3, c="limegreen", label="Euler Method", s=50)
    ax4.scatter(t_st, r3, c="purple", label="Runge Kutta order 2")
    ax4.set(xlabel="Time (seconds)", ylabel="Terminal Velocity (v)", title="Numerical Method Graph")
    ax5.scatter(t_st, e_3a, c="limegreen", label="Absolute Error in Euler Method")
    ax5.scatter(t_st, e_3b, c="purple", label="Absolute Error in RK2")
    ax5.set(xlabel="Time (seconds)", ylabel="Absolute Error", title="Absolute Error Graph")
    ax.grid(ls="--")
    ax1.grid(ls="--")
    ax2.grid(ls="--")
    ax3.grid(ls="--")
    ax4.grid(ls="--")
    ax5.grid(ls="--")
    ax.legend()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    fig.suptitle("Q3A: RADIOACTIVE DECAY")
    fig1.suptitle("Q3A: RADIOACTIVE DECAY")

    fig2.suptitle("Q3B: RC CIRCUIT")
    fig3.suptitle("Q3B: RC CIRCUIT")
    fig4.suptitle("Q3C: STOKES' LAW")
    fig5.suptitle("Q3C: STOKES' LAW")
    plt.show()

def Table(t_rd, t_rc, t_st, e1, e2, e3, r1, r2, r3, e_1a, e_1b, e_2a, e_2b, e_3a, e_3b, a1, a2, a3):
    Table1 = PrettyTable(['Sr No.', 'Time / Half life', 'No. of nuclei(Euler_Method)', 'No. of nuclei(RK2 Method)', 'No. of nuclei(Analytical)', 'Abs Error(Euler Method)', 'Abs Error(RK2 Method)']) 
    Table2 = PrettyTable(['Sr No.', 'Time(seconds)', 'Voltage(Euler Method)', 'Voltage(RK2 Method)', 'Voltage(Analytical)', 'Abs Error(Euler Method)', 'Abs Error(RK2 Method)'])
    Table3 = PrettyTable(['Sr No.', 'Time(seconds)', 'Velocity(Euler Method)', 'Velocity(RK2 Method)', 'Velocity(Analytical)', 'Abs Error(Euler Method)', 'Abs Error(RK2 Method)'])
    for i in range(len(t_r)):
        Table1.add_row([i + 1, t_rd[i], e1[i], r1[i], a1[i], e_1a[i], e_1b[i]])
        Table2.add_row([i + 1, t_rc[i], e2[i], r2[i], a2[i], e_2a[i], e_2b[i]])
        Table3.add_row([i + 1, t_st[i], e3[i], r3[i], a3[i], e_3a[i], e_3b[i]])
    print("\nQ3A: RADIOACTIVE DECAY\n", Table1, "\n")
    print("Q3B: RC CIRCUIT\n", Table2, "\n")
    print("Q3C: STOKES' LAW\n", Table3, "\n")

"Calling Functions for Calculations"
e1 = Euler_Method(h[0], c[0], τ[0], N_r)
e2 = Euler_Method(h[1], c[1], τ[1], N_rc)
e3 = Euler_Method(h[2], c[2], τ[2], N_st)
r1 = RK2_Method(h[0], c[0], τ[0], N_r)
r2 = RK2_Method(h[1], c[1], τ[1], N_rc)
r3 = RK2_Method(h[2], c[2], τ[2], N_st)
a1 = analytic(c[0], t_r, τ[0])
a2 = analytic(c[1], t_rc, τ[1])
a3 = analytic(c[2], t_st, τ[2])
e_1a = Absolute_Error(a1, e1); e_1b = Absolute_Error(a1, r1)
e_2a = Absolute_Error(a2, e2); e_2b = Absolute_Error(a2, r2)
e_3a = Absolute_Error(a3, e3); e_3b = Absolute_Error(a3, r3)
Table(t_rd, t_rc, t_st, e1, e2, e3, r1, r2, r3, e_1a, e_1b, e_2a, e_2b, e_3a, e_3b, a1, a2, a3)
Graph(t_rd, t_rc, t_st, e1, e2, e3, r1, r2, r3, e_1a, e_1b, e_2a, e_2b, e_3a, e_3b)