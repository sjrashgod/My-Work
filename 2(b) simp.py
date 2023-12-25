from scipy.integrate import quad


def integrand(x, s, t):
    return s + t + (x ** 2)

def No_intervals(n):
    N = 2 * n
    return N


def step_size(a, b, N):
    h = (b - a) / N
    return h


def intervals(a, h, N):
    list_1 = []
    list_1.append(a)
    for i in range(N):
        a = a + h
        list_1.append(a)
    return list_1


def func(list_1):
    list_2 = []
    for i in (list_1):
        y = i ** 2
        list_2.append(y)
    return list_2


def simpson_method(list_2, h):
    f2 = 0
    f3 = 0
    f1 = list_2[0] + list_2[len(list_2) - 1]
    for i in range(1, len(list_2) - 1):
            if i % 2 == 0:
                f2 = f2 + list_2[i]
            else:
                f3 = f3 + list_2[i]
    int_simp = h / 3 * (f1 + (4 * (f3)) + (2 * (f2)))

    return int_simp


def error(simpson, I):
    e1 = I[0] - simpson
    return e1


if __name__ == "__main__":
    n = int(input("Enter the no. of intervals, n : "))
    a = float(input("Enter the value of lower limit, a : "))
    b = float(input("Enter the value of upper limit, b : "))
    s = 0
    t = 0
    N = No_intervals(n)
    h = step_size(a, b, N)
    list_1 = intervals(a, h, N)
    list_2 = func(list_1)
    simpson = simpson_method(list_2, h)
    I = quad(integrand, 0, 1, args=(s, t))
    e = error(simpson, I)
    print("Analytical value of integration is", I[0])
    print("Numerical value of integration using simpson method is", simpson)
    print("Error in integration is", e)

