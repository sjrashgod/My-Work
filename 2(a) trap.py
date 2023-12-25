import numpy as np
from scipy.integrate import quad


def integrand(x, s, t):
    return s + t + (x ** 2)


def step_size(a, b, n):
    h = (b - a) / n
    return h


def intervals(a, h, n):
    list_1 = []
    list_1.append(a)
    for i in range(n):
        a = a + h
        list_1.append(a)
    return list_1


def func(list_1):
    list_2 = []
    for i in (list_1):
        y = i ** 2
        list_2.append(y)
    return list_2


def trapezoid_method(list_2, h):
    f2 = 0
    f1 = (list_2[0] + list_2[len(list_2) - 1]) / 2
    for i in range(1, len(list_2) - 1):
        f2 = f2 + list_2[i]
    int_trap = h * (f1 + f2)
    return int_trap


def error(trapezoid, I):
    e1 = I[0] - trapezoid
    return e1

if __name__ == "__main__":
    n = int(input("Enter the no. of intervals, N : "))
    a = float(input("Enter the value of lower limit, a : "))
    b = float(input("Enter the value of upper limit, b : "))
    s = 0
    t = 0
    h = step_size(a, b, n)
    list_1 = intervals(a, h, n)
    list_2 = func(list_1)
    trapezoid = trapezoid_method(list_2, h)
    I = quad(integrand, 0, 1, args=(s, t))
    e = error(trapezoid, I)
    print("Analytical value of integration is", I[0])
    print("Numerical value of integration using trapezoidal method is", trapezoid)
    print("Error in integration is", e)
