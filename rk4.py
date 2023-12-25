import numpy as np
import matplotlib.pyplot as plt


def func(y, x, consts=None):

    # Equations
    dx = y + x - x ** 3
    dy = -x

    # dydt = [First Eqn, Second Eqn, ...]
    dydt = [dx, dy]

    return dydt


def rk4(x0, y0, b, a, N,  h=None):
    # When h is not given directly
    if (h == None):
        h = (b - a) / N
    yarr = []
    yin = y0
    xin = x0

    for i in range(0,15, h):
        yarr.append(yin)

        k1 = [h * ele for ele in func(yin, xin, consts)]

        yn = [e1 + e2 / 2 for (e1, e2) in zip(yin, k1)]
        xn = [e1 + h / 2 for e1 in xin]
        k2 = [h * ele for ele in func(yn, xn, consts)]

        yn = [e1 + e2 / 2 for (e1, e2) in zip(yin, k2)]
        k3 = [h * ele for ele in func(yn, xn, consts)]

        yn = [e1 + e2 for (e1, e2) in zip(yin, k3)]
        xn = [e1 + h for e1 in xin]
        k4 = [h * ele for ele in func(yn, xn, consts)]

        yf = [ini_y + (e1 + 2 * (e2 + e3) + e4) / 6 for (ini_y, e1, e2, e3, e4) in zip(yin, k1, k2, k3, k4)]

        yin = yf
        xin = [e1 + h / 2 for e1 in xn]

    yarr = np.array(yarr).reshape(-1, 4)

    return (yarr)


if __name__ == '__main__':
    x0 = 0
    y0 = -1, -2, -3, -4

    b = 15
    a = 0
    N = 100