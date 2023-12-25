from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

"Here the test function is defined for analytical purpose"
def integrand(x):
    return x**2

"Function to define step size"
def step_size(a, b, n):
    h = (b - a) / n
    return h

"Function which produces a list of sub-intervals"
def intervals(a, h, n):
    list_1 = []
    list_1.append(a)
    for i in range(n):
        a = a + h
        list_1.append(a)
    return list_1

"Function which produces a list of values of each sub-interval"
def func(list_1):
    list_2 = []
    for i in (list_1):
        y = f(i)
        list_2.append(y)
    return list_2

#Q2(a) & Q3(b)
"Here the general Trapezoidal Method is defined"
def trapezoid_method(list_2, h):
    f2 = 0
    f1 = (list_2[0] + list_2[len(list_2) - 1]) / 2
    for i in range(1, len(list_2) - 1):
        f2 = f2 + list_2[i]
    int_trap = h * (f1 + f2)
    return int_trap

#Q2(b) & Q3(b)
"Here the general Simpson Method is defined"
def simpson_method(list_2,h,n):
    f2,f3 = 0,0
    if n%2 == 0:
        f1 = list_2[0] + list_2[len(list_2) - 1]
        for i in range(1, len(list_2) - 1):
            if i%2 == 0:
                f2 = f2 + list_2[i]
            else:
                f3 = f3 + list_2[i]
        int_simp = h / 3 * (f1 + (4 * (f3)) + (2 * (f2)))
    else:
        int_simp = None
    return int_simp

#Q2(c) & Q3(b)
"Function for computing error in integration"
def error(trapezoid,simpson, Int, n):
    list_3 = []
    if n % 2 == 0:
        e1 = abs(Int[0] - simpson)
        list_3.append(e1)
    else:
        e1 = None
        list_3.append(e1)
    e2 = abs(Int[0] - trapezoid)
    list_3.append(e2)
    return list_3

#Q2(d)
"Plotting for Q2(d)"
def log_plot(list_h, list_trapezoidal, list_simpson):
    plt.scatter(list_h, list_trapezoidal, c="g", label = "Trapezoidal")
    plt.scatter(list_h, list_simpson, c="r", label = "Simpson")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Log-plot of I(h) vs h")
    plt.legend()
    plt.show()

#Q3(a)
"Defining step size specifically for Q3(a)"
def q3_step_size(x,y,N):
    step = (y-x)/N
    return step

"Defining trapezoidal method specifically for Q3(a)"
def trap_method(step,V):
    z2 = 0
    z1 = V[0] + V[10]
    for i in range(1,len(V)-1):
        z2 = z2 + V[i]

    i_trap = step/2 * (z1 + 2*z2)
    return i_trap

"Defining simpson method specifically for Q3(a)"
def simp_method(step,V):
    z2 = 0
    z3 = 0
    z1 = V[0] + V[10]
    if (N%2)==0:
        for i in range(1,len(V)-1):
            if i % 2 == 0:
                z2 = z2 + V[i]
            else:
                z3 = z3 + V[i]
        i_simp = step/3 * (z1 + 4 * z3 + 2 * z2)
        return i_simp
    else:
        print("simpson method not applicable")

"Plotting for Q3(a)"
def plot(V, I):
    plt.style.use("seaborn")
    plt.scatter(I,V, c = "blue")
    plt.title("Q3(a) Current vs Voltage")
    plt.xlabel("Current (mA)")
    plt.ylabel("Voltage (V)")
    plt.show()

if __name__ == "__main__":
    f = eval("lambda x:" + input("Enter the function (test function = x**2) : "))
    n = int(input("Enter the no. of intervals, n : "))
    a = float(input("Enter the value of lower limit, a : "))
    b = float(input("Enter the value of upper limit, b : "))
    s = 0
    t = 0
    h = step_size(a, b, n)
    list_1 = intervals(a, h, n)
    list_2 = func(list_1)
    trapezoid = trapezoid_method(list_2, h)
    simpson = simpson_method(list_2, h, n)
    Int = quad(integrand, 0, 1)
    e = error(trapezoid, simpson, Int, n)
    print("Analytical value of integration is", Int[0])
    print("Q2 (a) & Q3 (b) Numerical value of integration using trapezoidal method is", trapezoid)
    print("Q2 (b) & Q3 (b) Numerical value of integration using simpson method is", simpson)
    print("Q2 (c) & Q3(b) Error in integration [error in trapezoid method, error in simpson method] is", e)
    list_h = []
    list_simpson = []
    list_trapezoidal = []
    for n in range(2, 101): #Q2(d)
        a = 0  # lower limit
        b = 1  # upper limit
        h = step_size(a, b, n)
        list_1 = intervals(a, h, n)
        list_2 = func(list_1)
        simpson = simpson_method(list_2, h, n)
        trapezoidal = trapezoid_method(list_2, h)
        list_h.append(h)
        list_simpson.append(simpson)
        list_trapezoidal.append(trapezoidal)
    V = np.array([0.0, 0.5, 2.0, 4.05, 8.0, 12.5, 18.0, 24.5, 32.0, 40.5, 50.0])
    I = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    x = I[0]
    y = I[10]
    N = len(I) - 1
    step = q3_step_size(x,y, N)
    i_trap = trap_method(step, V)
    i_simp = simp_method(step, V)
    print("Q3 (a) Step size : ", step)
    print("Q3 (a) Estimated power delivered (trapezoidal method) is : ", i_trap)
    print("Q3 (a) Estimated power delivered (simpson method) is : ", i_simp)
    log_plot(list_h, list_trapezoidal, list_simpson)
    plot(V, I)