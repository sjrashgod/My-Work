import numpy as np
from MyIVP import elr, rk2, rk4
import matplotlib.pyplot as plt

def Func(yin, xval):
    y1, y2, y3 = yin
    f1 = y2 - y3 + xval
    f2 = 3*xval**2
    f3 = y2 + np.exp(-xval)
    eqn = [f1, f2, f3]
    return eqn

def outputs(Func, initial_Cond, a, b, n):
    res_elr = elr(Func,initial_Cond,a,b, n)
    res_rk2 = rk2(Func,initial_Cond,a,b, n)
    res_rk4 = rk4(Func,initial_Cond,a,b, n)
    x_axis = np.linspace(a, b, n)

    return res_elr, res_rk2, res_rk4, x_axis

def table(data,head,Title):
    print(Title)
    line='_'*len(head)*12+'____'
    for i in head:
        print("{0:^12}".format(i),end=" ")
    print("\n",line)
    for row in data:
        for val in row:
            print("{0:^12.2}".format(val), end=" ")
        print("\n")

def graph(x_axis, X, title):
    for i in range(len(X)):
        plt.plot(x_axis, X[i], label = "Y" + str(i))  
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("$y_i$")
    plt.legend()
    plt.grid()

def graph_analytic(x_axis, Analytic):
    for g, i in zip(Analytic, range(len(Analytic))):
        plt.plot(x_axis, g(x_axis), label = "Analytic_Y" + str(i), ls="--", c = "black") 
    plt.legend()

def plot3(X, x_axis, N):
    for i in range(len(X)):
        plt.plot(x_axis, X[i], label = "$N={0}$".format(N))

if __name__ == '__main__':
    initial_Cond = [1, 1, -1]
    a = 0
    b = 1
    n = 100

    analytic_sol_1 = lambda x : -0.05*x**5 + 0.25*x**4 + x +2 - np.exp(-x)
    analytic_sol_2 = lambda x :  x**3 +1
    analytic_sol_3 = lambda x :  0.25*x**4 + x - np.exp(-x)
    Analytic = [analytic_sol_1, analytic_sol_2, analytic_sol_3]

    X, Y, Z, x_axis = outputs(Func, initial_Cond, a, b, n)

    EUL_data = np.column_stack((x_axis, *X))
    RK2_data = np.column_stack((x_axis, *Y))
    RK4_data = np.column_stack((x_axis, *Z))

    '''Table'''
    heading = ["x","Y1","Y2","Y3"]
    table(EUL_data, heading, "\nEULER METHOD FOR N=100")
    table(RK2_data, heading, "\nRK2 METHOD FOR N=100")
    table(RK4_data, heading, "\nRK4 METHOD FOR N=100")

    '''Graphs'''
    graph(x_axis, X,"\nEULER METHOD FOR N=100")
    graph_analytic(x_axis, Analytic)
    plt.show()

    graph(x_axis, Y,"\nRK2 METHOD FOR N=100")
    graph_analytic(x_axis, Analytic)
    plt.show()

    graph(x_axis, Z,"\nRK4 METHOD FOR N=100")
    graph_analytic(x_axis, Analytic)
    plt.show()

    N_arr = [10**k for k in range(5)]
    elr_err_1 = []; elr_err_2 = []; elr_err_3 = []
    rk2_err_1 = []; rk2_err_2 = []; rk2_err_3 = []
    rk4_err_1 = []; rk4_err_2 = []; rk4_err_3 = []

    for i in range(0, 5):
        N = N_arr[i]
        X, Y, Z, t = outputs(Func, initial_Cond, a, b, n)
        f1 = analytic_sol_1(t)
        f2 = analytic_sol_2(t)
        f3 = analytic_sol_3(t)
        elr_err_1.append(max(np.array(f1)-np.array(X[0])))
        elr_err_2.append(max(np.array(f2)-np.array(X[1])))
        elr_err_3.append(max(np.array(f3)-np.array(X[2])))
        plot3(X, t, N)

    graph_analytic(x_axis, Analytic)   
    plt.title("Euler Method for different N")
    plt.xlabel("x")
    plt.ylabel("$y_i$")
    plt.legend()
    plt.grid()
    plt.show()

    for i in range(0, 5):
        N = N_arr[i]
        X, Y, Z, t = outputs(Func, initial_Cond, a, b, n)
        f1 = analytic_sol_1(t)
        f2 = analytic_sol_2(t)
        f3 = analytic_sol_3(t)
        rk2_err_1.append(max(np.array(f1)-np.array(Y[0])))
        rk2_err_2.append(max(np.array(f2)-np.array(Y[1])))
        rk2_err_3 . append(max(np.array(f3)-np.array(Y[2])))
        plot3(Y, t, N)

    graph_analytic(x_axis, Analytic)
    plt.title("RK2 Method for different N")
    plt.xlabel("x")
    plt.ylabel("$y_i$")
    plt.legend()
    plt.grid()
    plt.show()

    for i in range(0, 5):
        N = N_arr[i]
        X, Y, Z, t = outputs(Func, initial_Cond, a, b, n)
        plot3(Z, t, N)
        f1 = analytic_sol_1(t)
        f2 = analytic_sol_2(t)
        f3 = analytic_sol_3(t)
        rk4_err_1.append(max(np.array(f1)-np.array(Z[0])))
        rk4_err_2.append(max(np.array(f2)-np.array(Z[1])))
        rk4_err_3.append(max(np.array(f3)-np.array(Z[2])))

    graph_analytic(x_axis, Analytic)   
    plt.title("RK4 Method for different N")
    plt.xlabel("x")
    plt.ylabel("$y_i$")
    plt.legend()
    plt.grid()
    plt.show()

    print("\n")
    data = np.column_stack((N_arr, elr_err_1, rk2_err_1, rk4_err_1, elr_err_2, rk2_err_2, rk4_err_2, elr_err_3, rk2_err_3, rk4_err_3))
    
    head = ["N", "E_y1(Euler)", "E_y1(RK2)", "E_y1(RK4)", "E_y2(Euler)", "E_y2(RK2)", "E_y2(RK4)", "E_y3(Euler)", "E_y3(RK2)", "E_y3(RK4)"]
    table(data, head, "\nTable for N and E= max(|y_ana -y_num|) values for y0,y1 and y2 for all three methods")