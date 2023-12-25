import matplotlib.pyplot as plt
import numpy as np
from MyIntegration import trapz, simps, gaussquad
from sympy.abc import x
from sympy import sin, cos, lambdify

def FourierCoeff(expr, N, L, method, ftype = -1, tol = 1, n = 100, m = 1, pw = False):
    a0 = 0
    an = []
    bn = []

    if (method == 'gaussquad'):
        n = 5
        m = 5
    techniques = {'trapz': trapz,
                'simps': simps,
                'gaussquad': gaussquad}
    int_tech = techniques[method]

    if (ftype == 0):    # Even Function
        bn = np.zeros(shape = [1, N])
        a0 = int_tech(expr, a = 0, b = L, n = n, m = m, tol = tol)[0]/(L)    # a0/2

        for i in range(1, N+1):
            func = "(" + expr + ")" + "*" + "np.cos({:}*(x*np.pi/{:}))".format(str(i), str(L))
            a_coeff = int_tech(func, a = 0, b = L, n = n, m = m, tol = tol)[0]
            an.append(2*a_coeff/L)
    
        if (pw == True):
            an = np.array(an)/2       # /2 because we require int-0 to L and not int-(-L) to L.
        
        an = np.array(an)

    elif (ftype == 1):    # Odd Function
        an = np.zeros(shape = [1, N])

        for i in range(1, N+1):
            func = "(" + expr + ")" + "*" + "np.sin({:}*(x*np.pi/{:}))".format(str(i), str(L))
            b_coeff = int_tech(func, a = 0, b = L, n = n, m = m, tol = tol)[0]
            bn.append(2*b_coeff/L)
        
        if (pw == True):
            a0 = int_tech(expr, a = 0, b = L, n = n, m = m, tol = tol)[0]/(2*L) # a0/2
            bn = np.array(bn)/2       # /2 because we require int-0 to L and not int-(-L) to L.

        bn = np.array(bn)

    elif (ftype == -1):    # Neither Odd nor Even Function
        a0 = int_tech(expr, a = -L, b = L, n = n, m = m, tol = tol)[0]/(2*L)    # a0/2

        if (pw == True):
            a0 = a0/2

            for i in range(1, N+1): 
                func_cos = "(" + expr + ")" + "*" + "np.cos({:}*(x*np.pi/(2*{:})))".format(str(i), str(L))
                a_coeff = int_tech(func_cos, a = -L, b = L, n = n, m = m, tol = tol)[0]
                an.append(a_coeff/(2*L))

                func_sin = "(" + expr + ")" + "*" + "np.sin({:}*(x*np.pi/(2*{:})))".format(str(i), str(L))
                b_coeff = int_tech(func_sin, a = -L, b = L, n = n, m = m, tol = tol)[0]
                bn.append(b_coeff/(2*L))
        
        else:
            for i in range(1, N+1): 
                func_cos = "(" + expr + ")" + "*" + "np.cos({:}*(x*np.pi/{:}))".format(str(i), str(L))
                a_coeff = int_tech(func_cos, a = -L, b = L, n = n, m = m, tol = tol)[0]
                an.append(a_coeff/L)

                func_sin = "(" + expr + ")" + "*" + "np.sin({:}*(x*np.pi/{:}))".format(str(i), str(L))
                b_coeff = int_tech(func_sin, a = -L, b = L, n = n, m = m, tol = tol)[0]
                bn.append(b_coeff/L)
        
        an = np.array(an)
        bn = np.array(bn)
        
    return(a0, np.array(an.flatten()), np.array(bn.flatten()))


if __name__  == '__main__':
    print("\nName: Samarth Jain\tRoll No. : 2020PHY1089\nPartner: Aashmeet Kaur\tRoll No.: 2020PHY1138\n")

    n = [1, 2, 5, 10, 20]
    xx = np.linspace(-2.5, 2.5, 50)

    # # Function 1

    f1 = np.piecewise(xx, [ ((-3<xx) & (xx<-2)), ((-2<xx) & (xx<-1)), ((-1<xx) & (xx<0)), ((0<xx) & (xx<1)), ((1<xx) & (xx<2)), ((2<xx) & (xx<3))], [0, 1, 0, 1, 0, 1])
    dataf1 = []
    outf1 = []
    errf1 = []
    partialsum_f1 = []
    for i in n:
        a0, an, bn = FourierCoeff(expr = "1", N = i, L = 1, method = 'simps', ftype = 1, pw = True)
        partialsumf1 = a0
        for j in range(len(an)):
            partialsumf1 += bn[j]*sin(int(j+1)*np.pi*x) + an[j]*cos(int(j+1)*np.pi*x)
        partialsum_f1.append(lambdify(x, partialsumf1))
    
    plt.figure("Piecewise Function 1")
    plt.plot(xx, f1, label = "Function 1")
    for (p, q) in zip(partialsum_f1, n):
        plt.plot(xx, p(xx), marker = '.', linewidth = 0.5, label = "N = %d"%q)
        dataf1.append(p(xx))
        outf1.append(p(-0.5))
        outf1.append(p(0))
        outf1.append(p(0.5))
        errf1.append(np.abs(p(-0.5) - np.zeros(shape = [1, 50])))
        errf1.append(np.abs(p(0) - 0.5*np.ones(shape = [1, 50])))
        errf1.append(np.abs(p(0.5) - np.ones(shape = [1, 50])))


    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Samarth Jain and Aashmeet Kaur\nFourier Approximation of Piecewise Function (1)")
    plt.legend()
    data_f1 = np.column_stack((xx, *dataf1))
    np.savetxt(r'C:\Users\jains\Desktop\Function1.txt', data_f1, header = "x, 1, 2, 5, 10, 20")
    data_out_f1 = np.column_stack((*outf1, *errf1))
    np.savetxt(r'C:\Users\jains\Desktop\Output_Function_1.txt', data_out_f1, header = "n=1 fr(-0.5), fr(0), fr(0.5), n=2 fr(-0.5), fr(0), fr(0.5), n=5 fr(-0.5), fr(0), fr(0.5), n=10 fr(-0.5), fr(0), fr(0.5), n=20 fr(-0.5), fr(0), fr(0.5), n=1 err(-0.5), err(0), err(0.5), n=2 err(-0.5), err(0), err(0.5), n=5 err(-0.5), err(0), err(0.5), n=10 err(-0.5), err(0), err(0.5), n=20 err(-0.5), err(0), err(0.5)")


    # Function 2

    f2 = np.piecewise(xx, [ ((-3<xx) & (xx<-2.5)), ((-2.5<xx) & (xx<-1.5)), ((-1.5<xx) & (xx<-1)), ((-1<xx) & (xx<-0.5)), ((-0.5<xx) & (xx<0.5)), ((0.5<xx) & (xx<1)), ((1<xx) & (xx<1.5)), ((1.5<xx) & (xx<2.5)), ((2.5<xx) & (xx<3))], [0, 1, 0, 0, 1, 0, 0, 1, 0])
    dataf2 = []
    outf2 = []
    errf2 = []
    partialsum_f2 = []
    for i in n:
        a0, an, bn = FourierCoeff(expr = "1", N = i, L = 0.5, method = 'simps', ftype = -1, pw = True)
        partialsumf2 = a0
        for j in range(len(an)):
            partialsumf2 += bn[j]*sin(int(j+1)*np.pi*x) + an[j]*cos(int(j+1)*np.pi*x)
        partialsum_f2.append(lambdify(x, partialsumf2))

    plt.figure("Piecewise Function 2")
    plt.plot(xx, f2, label = "Function 2")
    for (p, q) in zip(partialsum_f2, n):
        plt.plot(xx, p(xx), marker = '.', linewidth = 0.5, label = "N = %d"%q)
        dataf2.append(p(xx))
        outf2.append(p(-0.5))
        outf2.append(p(0))
        outf2.append(p(0.5))
        errf2.append(np.abs(p(-0.5) - np.zeros(shape = [1, 50])))
        errf2.append(np.abs(p(0) - 0.5*np.ones(shape = [1, 50])))
        errf2.append(np.abs(p(0.5) - np.ones(shape = [1, 50])))
    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Samarth Jain and Aashmeet Kaur\nFourier Approximation of Piecewise Function (2)")
    plt.legend(loc = "lower right")
    data_f2 = np.column_stack((xx, *dataf2))
    np.savetxt(r'C:\Users\jains\Desktop\Function2.txt', data_f2, header = "x, 1, 2, 5, 10, 20")
    data_out_f2 = np.column_stack((*outf2, *errf2))
    np.savetxt(r'C:\Users\jains\Desktop\Output_Function_2.txt', data_out_f2, header = "n=1 fr(-0.5), fr(0), fr(0.5), n=2 fr(-0.5), fr(0), fr(0.5), n=5 fr(-0.5), fr(0), fr(0.5), n=10 fr(-0.5), fr(0), fr(0.5), n=20 fr(-0.5), fr(0), fr(0.5), n=1 err(-0.5), err(0), err(0.5), n=2 err(-0.5), err(0), err(0.5), n=5 err(-0.5), err(0), err(0.5), n=10 err(-0.5), err(0), err(0.5), n=20 err(-0.5), err(0), err(0.5)")
    
    # Function 3

    f3 = np.piecewise(xx, [ ((-3<xx) & (xx<-2)), ((-2<xx) & (xx<-1)), ((-1<xx) & (xx<0)), ((0<xx) & (xx<1)), ((1<xx) & (xx<2)), ((2<xx) & (xx<3))], [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    dataf3 = []
    outf3 = []
    errf3 = []    
    partialsum_f3 = []
    for i in n:
        a0, an, bn = FourierCoeff(expr = "0.5", N = i, L = 1, method = 'simps', ftype = 1)
        partialsumf3 = a0
        for j in range(len(an)):
            partialsumf3 += bn[j]*sin(int(j+1)*np.pi*x) + an[j]*cos(int(j+1)*np.pi*x)
        partialsum_f3.append(lambdify(x, partialsumf3))

    plt.figure("Piecewise Function 3")
    plt.plot(xx, f3, label = "Function 3")
    for (p, q) in zip(partialsum_f3, n):
        plt.plot(xx, p(xx), marker = '.', linewidth = 0.5, label = "N = %d"%q)
        dataf3.append(p(xx))
        outf3.append(p(-0.5))
        outf3.append(p(0))
        outf3.append(p(0.5))
        errf3.append(np.abs(p(-0.5) - np.zeros(shape = [1, 50])))
        errf3.append(np.abs(p(0) - 0.5*np.ones(shape = [1, 50])))
        errf3.append(np.abs(p(0.5) - np.ones(shape = [1, 50])))
    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Samarth Jain and Aashmeet Kaur\nFourier Approximation of Piecewise Function (3)")
    plt.legend()
    data_f3 = np.column_stack((xx, *dataf3))
    np.savetxt(r'C:\Users\jains\Desktop\Function3.txt', data_f3, header = "x, 1, 2, 5, 10, 20")
    data_out_f3 = np.column_stack((*outf3, *errf3))
    np.savetxt(r'C:\Users\jains\Desktop\Output_Function_3.txt', data_out_f3, header = "n=1 fr(-0.5), fr(0), fr(0.5), n=2 fr(-0.5), fr(0), fr(0.5), n=5 fr(-0.5), fr(0), fr(0.5), n=10 fr(-0.5), fr(0), fr(0.5), n=20 fr(-0.5), fr(0), fr(0.5), n=1 err(-0.5), err(0), err(0.5), n=2 err(-0.5), err(0), err(0.5), n=5 err(-0.5), err(0), err(0.5), n=10 err(-0.5), err(0), err(0.5), n=20 err(-0.5), err(0), err(0.5)")

    plt.show()