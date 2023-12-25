import MyIntegration as mi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import var,sympify 
from sympy.utilities.lambdify import lambdify

# Name => Ishmeet Singh   Roll no. => 2020PHY1221
# Patner's Name => Sarthak Jain   Roll no. => 2020PHY1201

def func1(eps, x0 = 0):
    result = "exp(-(x-{:})**2/(2*({:})**2)) / (2*pi*({:})**2)**(1/2)".format(x0, eps, eps)
    return(result)

def func2(eps, x0 = 0):
    result = "{:}/(pi*((x-{:})**2 + ({:})**2))".format(eps, x0, eps)
    return((result))

if __name__ == '__main__':
    x_axis = np.linspace(0, 2, 1000)
    epsilon = [0.4/2**i for i in range(1, 6)]

    """Dirac-delta Plots: 2(a) i"""
    for j in epsilon:
        plt.figure("Gaussian")
        x = var("x")
        expr1 = sympify(func1(eps = j, x0 = 1))
        function1 = lambdify(x,expr1)
        plt.plot(x_axis, function1(x_axis), label = "$\epsilon = ${:.3e}".format(j))
        plt.figure("Lorentzian")
        expr2 = sympify(func2(eps = j, x0 = 1))
        function2 = lambdify(x,expr2)
        plt.plot(x_axis, function2(x_axis), label = "$\epsilon = ${:.3e}".format(j))
    
    plt.figure("Gaussian")
    plt.ylabel("$\delta_{\epsilon}(x-1)$")
    plt.xlabel("x Axis")
    plt.title("Dirac-delta function as an approximation of Gaussian Distribution\ncentered at $a = 1$")
    plt.legend()

    plt.figure("Lorentzian")
    plt.ylabel("$\delta_{\epsilon}(x-1)$")
    plt.xlabel("x Axis")
    plt.title("Dirac-delta function as an approximation of Lorentz/Cauchy Distribution\ncentered at $a = 1$")
    plt.legend()

    """Dirac-delta Properties: 2(a) ii"""
    # Analytic Results
    x_axis2 = np.linspace(-2, 2, 1000)
    res1 , res2 , res3 = np.ones(len(epsilon)) , np.ones(len(epsilon)), np.ones(len(epsilon))
    # Dirac-delta convolutions using Simpsons 1/3
    Integral1 = [[] for i in range(4)]
    Integral2 = [[] for i in range(4)]
    Integral3 = [[] for i in range(4)]
    for j in epsilon:
        """Integral 1"""
        int11 = mi.MySimp(func1(eps = j),2,-2,1000)
        int12 = mi.MySimp(func2(eps = j),2,-2,1000)
        Herm_int11 = mi.MyHermiteQuad(expression = "exp(x**2)*"+func1(eps = j), n = 50)
        Herm_int12 = mi.MyHermiteQuad(expression = "exp(x**2)*"+func2(eps = j), n = 50)
        Integral1[0].append(int11)
        Integral1[1].append(int12)
        Integral1[2].append(Herm_int11)
        Integral1[3].append(Herm_int12)

        """Integral 2"""
        int21 = mi.MySimp("((x+1)**2)*"+func1(eps = j),2,-2,1000)
        int22 = mi.MySimp("((x+1)**2)*"+func2(eps = j),2,-2,1000)
        Herm_int21 = mi.MyHermiteQuad("exp(x**2)*((x+1)**2)*"+func1(eps = j), n = 50)
        Herm_int22 = mi.MyHermiteQuad("exp(x**2)*((x+1)**2)*"+func2(eps = j), n = 50)
        Integral2[0].append(int21)
        Integral2[1].append(int22)
        Integral2[2].append(Herm_int21)
        Integral2[3].append(Herm_int22)

        """Integral 3"""
        int31 = mi.MySimp("((3*x)**2)*"+func1(eps = j, x0 = -1/3),2,-2,1000)
        int32 = mi.MySimp("((3*x)**2)*"+func2(eps = j, x0 = -1/3),2,-2,1000)
        Herm_int31 = mi.MyHermiteQuad("exp(x**2)*((3*x)**2)*"+func1(eps = j), n = 50)
        Herm_int32 = mi.MyHermiteQuad("exp(x**2)*((3*x)**2)*"+func2(eps = j), n = 50)
        Integral3[0].append(int31)
        Integral3[1].append(int32)
        Integral3[2].append(Herm_int31)
        Integral3[3].append(Herm_int32)
    
    plt.figure("SHI1")
    plt.plot(epsilon, res1, color = "black", label = "Analytical Result")
    plt.scatter(epsilon, Integral1[0], label = "$\delta_{\epsilon}(x)$ as Gaussian Distribution - Simpson 1/3", zorder = 5)
    plt.scatter(epsilon, Integral1[1], label = "$\delta_{\epsilon}(x)$ as Lorentzian Distribution - Simpson 1/3", zorder = 5)
    plt.scatter(epsilon, Integral1[2], marker = "*", label = "$\delta_{\epsilon}(x)$ as Gaussian Distribution - Gauss-Hermite", zorder = 5)
    plt.scatter(epsilon, Integral1[3], marker = "*", label = "$\delta_{\epsilon}(x)$ as Gaussian Distribution - Gauss-Hermite", zorder = 5)
    plt.xlabel("$\epsilon$")
    plt.ylabel("y Axis")
    plt.title("INTEGRAL 1\n")
    plt.legend()

    plt.figure("SHI2")
    plt.plot(epsilon, res2, color = "black", label = "Analytical Result")
    plt.scatter(epsilon, Integral2[0], label = "$\delta_{\epsilon}(x)$ as Gaussian Distribution - Simpson 1/3", zorder = 5)
    plt.scatter(epsilon, Integral2[1], label = "$\delta_{\epsilon}(x)$ as Lorentzian Distribution - Simpson 1/3", zorder = 5)
    plt.scatter(epsilon, Integral2[2], marker = "*", label = "$\delta_{\epsilon}(x)$ as Gaussian Distribution - Gauss-Hermite", zorder = 5)
    plt.scatter(epsilon, Integral2[3], marker = "*", label = "$\delta_{\epsilon}(x)$ as Gaussian Distribution - Gauss-Hermite", zorder = 5)
    plt.xlabel("$\epsilon$")
    plt.ylabel("y Axis")
    plt.title("INTEGRAL 2\n")
    plt.legend()

    plt.figure("SHI3")
    plt.plot(epsilon, res3, color = "black", label = "Analytical Result")
    plt.scatter(epsilon, Integral3[0], label = "$\delta_{\epsilon}(x)$ as Gaussian Distribution - Simpson 1/3", zorder = 5)
    plt.scatter(epsilon, Integral3[1], label = "$\delta_{\epsilon}(x)$ as Lorentzian Distribution - Simpson 1/3", zorder = 5)
    plt.scatter(epsilon, Integral3[2], marker = "*", label = "$\delta_{\epsilon}(x)$ as Gaussian Distribution - Gauss-Hermite", zorder = 5)
    plt.scatter(epsilon, Integral3[3], marker = "*", label = "$\delta_{\epsilon}(x)$ as Gaussian Distribution - Gauss-Hermite", zorder = 5)
    plt.xlabel("$\epsilon$")
    plt.ylabel("y Axis")
    plt.title("INTEGRAL 3\n")
    plt.legend()

    plt.show()

    """Dirac-delta Data Tables: 2(b)"""
    print("\nIntegral 1\n")
    int1 = {"Epsilon": epsilon, 
            "Gaussian Dist. - Simpson 1/3": Integral1[0], 
            "Lorentzian Dist. - Simpson 1/3": Integral1[1],
            "Gaussian Dist. - Gauss-Hermite": Integral1[2], 
            "Lorentzian Dist. - Gauss-Hermite": Integral1[3]
            }
    df1 = pd.DataFrame(int1)
    df1.to_csv(r'F:\Ishu\Dirac delta\Integral1.csv')
    print(df1)

    print("\nIntegral 2\n")
    int2 = {"Epsilon": epsilon, 
            "Gaussian Dist. - Simpson 1/3": Integral2[0], 
            "Lorentzian Dist. - Simpson 1/3": Integral2[1],
            "Gaussian Dist. - Gauss-Hermite": Integral2[2], 
            "Lorentzian Dist. - Gauss-Hermite": Integral2[3]
            }
    df2 = pd.DataFrame(int2)
    df2.to_csv(r'F:\Ishu\Dirac delta\Integral2.csv')
    print(df2)

    print("\nIntegral 3\n")
    int3 = {"Epsilon": epsilon, 
            "Gaussian Dist. - Simpson 1/3": Integral3[0], 
            "Lorentzian Dist. - Simpson 1/3": Integral3[1],
            "Gaussian Dist. - Gauss-Hermite": Integral3[2], 
            "Lorentzian Dist. - Gauss-Hermite": Integral3[3]
            }
    df3 = pd.DataFrame(int3)
    df3.to_csv(r'F:\Ishu\Dirac delta\Integral3.csv')
    print(df3)