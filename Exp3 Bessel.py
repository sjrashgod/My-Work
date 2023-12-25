import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import lagrange

"Q2 (A) Lagrange Interpolation Basis Function"
def lagrange_interpolation(s, t, x):
    y = 0
    for i in range(len(t)):
        p = 1
        for j in range(len(t)):
            if j != i:
                p = p * ((x - s[j]) / (s[i] - s[j]))
        y = y + t[i] * p
    return y

"Q2 (B) Inverse Lagrange Interpolation"
def inverse_lagrange(s, t, y):
    x = 0
    for i in range(len(s)):
        p = 1
        for j in range(len(s)):
            if j != i:
                p = p * ((y - t[j]) / (t[i] - t[j]))
        x = x + s[i] * p
    return x

"Given Data sets for Q3 (A)"
beta = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.4, 2.6, 2.8, 3.0])
J_o = np.array([1.0, 0.99, 0.96, 0.91, 0.85, 0.76, 0.67, 0.57, 0.46, 0.34, 0.22, 0.11, 0.00, -0.10, -0.18, -0.26])

"Given Data sets for Q3 (B)"
I = np.array([2.81, 3.24, 3.80, 4.30, 4.37, 5.29, 6.03])
V = np.array([0.5, 1.2, 2.1, 2.9, 3.6, 4.5, 5.7])

beta_plt = np.linspace(beta[0], beta[-1], 1000)
V_plt = np.linspace(V[0], V[-1], 1000)
j_o_inverse = np.linspace(J_o[0], J_o[-1], 1000)
x1 = 2.3 #Given value of beta
x2 = 2.4 #Given value for Bessel Function (J_o)
y = 0.5 #Given value for detected photodetector voltage
poly = lagrange(beta, J_o)
poly1 = lagrange(V, I)
I_plt = [];
J_o_plt = [];
I_in = [];
J_o_in = [];
beta_inverse = [];
inter1 = [];
inter2 = [];
inter3 = []

"Graphing"
def graph(beta, J_o, I, V, beta_plt, j_o_inverse, V_plt, I_plt, J_o_plt, I_in, J_o_in, beta_inverse, inter1, inter2,
          inter3):
    plt.style.use("seaborn")
    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    ax.plot(beta_plt, J_o_plt,c="crimson",label="Bessel Function $J_o(\\beta)$")
    ax.scatter(inter1[0], inter1[1],c="purple",label="Interpolated Point $(\\beta = 2.3)$", s=70)
    ax.scatter(beta, J_o,c="darkorange",label="Given Data Points")
    ax.set(title="Q3 (A): Bessel Function",xlabel="$\\beta$",ylabel="$J_o(\\beta)$")
    ax1.plot(V_plt, I_plt,c="royalblue",label="V-I Curve")
    ax1.scatter(inter2[0], inter2[1],c="deeppink",label="Interpolated Point (V = 2.4)", s=70)
    ax1.scatter(V, I,c="orange",label="Given Data Points")
    ax1.set(title="Q3 (B): Linear Interpolation",xlabel="Photodetector Voltage (V)",ylabel="Incident Laser Intensity (I)")

    ax2.plot(beta_plt, J_o_in,c="crimson",label="Bessel Function $J_o(\\beta)$ [Scipy]")
    ax2.scatter(inter1[0], inter1[2],c="purple",label="Interpolated Point $(\\beta = 2.3)$", s=70)
    ax2.scatter(beta, J_o,c="darkorange",label="Given Data Points")
    ax2.set(title="Q3 (A) Bessel function (Inbuilt)",xlabel="$\\beta$",ylabel="$J_o(\\beta)$")
    ax3.plot(V_plt, I_in,c="royalblue",label="V-I curve [Scipy]")
    ax3.scatter(inter2[0], inter2[2],c="deeppink",label="Interpolated Point (V = 2.4)", s=70)
    ax3.scatter(V, I,c="orange",label="Given Data Points")
    ax3.set(title="Q3 (B): Linear Interpolation (Inbuilt)",xlabel="Photodetector Voltage (V)",ylabel="Incident Laser Intensity (I)")
    ax4.plot(j_o_inverse, beta_inverse,c="limegreen",label="Inverse Bessel Function")
    ax4.scatter(inter3[0], inter3[1],c="dodgerblue",label="Interpolated Point ${J_o(\\beta) = 0.5}$", s=70)
    ax4.scatter(J_o, beta,c="chocolate",label="Given Data Points")
    ax4.set(title="Q3 (A): Inverse Bessel function",xlabel="$J_o(\\beta)$",ylabel="$\\beta$")
    ax.legend()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.show()

for i in beta_plt:
    J_o_plt.append(lagrange_interpolation(beta, J_o, i))
for j in V_plt:
    I_plt.append(lagrange_interpolation(V, I, j))
for k in beta_plt:
    J_o_in.append(poly(k))
for l in V_plt:
    I_in.append(poly1(l))
for l1 in j_o_inverse:
    beta_inverse.append(inverse_lagrange(beta, J_o, l1))

print('\nQ3 (A) 1st Part: Value of the Bessel Function (J_o) for given beta = 2.3 is:', lagrange_interpolation(beta, J_o, x1))
print('\nQ3 (A) 1st Part: Value of the Bessel Function (J_o) for given beta = 2.3, using inbuilt function is:', poly(x1))
inter1.extend([x1, lagrange_interpolation(beta, J_o, x1), poly(x1)])
print('\nQ3 (A) 2nd Part: Beta value for given Bessel Function (J_o) = 0.5 is:', inverse_lagrange(beta, J_o, y))
inter3.extend([y, inverse_lagrange(beta, J_o, y)])
print('\nQ3 (B): Value of incident laser intensity for given detected photodetector voltage = 0.5 is:', lagrange_interpolation(V, I, x2))
print('\nQ3 (B):Value of incident laser intensity for given detected photodetector voltage = 0.5, using inbuilt function is:', poly1(x2))
inter2.extend([x2, lagrange_interpolation(V, I, x2), poly1(x2)])
graph(beta, J_o, I, V, beta_plt, j_o_inverse, V_plt, I_plt, J_o_plt, I_in, J_o_in, beta_inverse, inter1, inter2, inter3)