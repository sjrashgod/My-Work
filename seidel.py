import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from prettytable import PrettyTable

def check(a,n):
    check_list = []
    for i in range(0,n):
        ele_sum = 0
        for j in range(0,n):
            if i != j:
                ele_sum = abs(ele_sum + a[i,j])
        if abs(a[i,i]) >= ele_sum:
            check_list.append(1)
        else:
            check_list.append(0)
    return check_list

def gauss_seidel(a,b,epsilon,n,x):
    x1 = [x[0]] ; x2 = [x[1]] ; x3 = [x[2]] ; count_list = [0] ; count  = 0
    while True:
        count = count + 1
        count_list.append(count)
        for i in range(0,n):
            const = 0
            for j in range(0,n):
                if i != j:
                    const = const - a[i,j]*x[j]
            x[i] = 1/a[i,i]*(b[i] + const)
            if i == 0:
                x1.append(x[i])
            elif i == 1:
                x2.append(x[i])
            elif i == 2:
                x3.append(x[i])
        if count > 0 and abs(x1[-1] - x1[-2]) <= epsilon:
            break
    print("\nSolution of the system of equations using gauss seidel is:\n",np.reshape(x,(n,1)))
    return [count_list,x1,x2,x3]

def graph(sol,n):
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    ax[0].scatter(sol[0],sol[1],label = "$X_1$")
    ax[1].scatter(sol[0],sol[2],label = "$X_2$",c = "orange")
    ax[2].scatter(sol[0],sol[3],label = "$X_3$",c = "g")
    ax[0].set(xlabel = "Number of Iteration",ylabel = "$X_1$")
    ax[1].set(xlabel = "Number of Iteration",ylabel = "$X_2$")
    ax[2].set(xlabel = "Number of Iteration",ylabel = "$X_3$")
    ax[0].grid(ls = "--")
    ax[1].grid(ls = "--")
    ax[2].grid(ls = "--")
    ax[0].legend(loc = "lower right")
    ax[1].legend(loc = "upper right")
    ax[2].legend(loc = "lower right")
    fig.suptitle("CONVERGENCE GRAPH")
    plt.show()

def table(sol):
    table1 = PrettyTable(["Iteration","X_1","X_2","X_3"])
    for i in range(len(sol[0])):
        table1.add_row([sol[0][i],sol[1][i],sol[2][i],sol[3][i]])
    print("\nValue After every iteration:\n",table1)

if __name__ == "__main__":
    a = np.array([[4,-1,0],[-1,5,-1],[0,-1,3]],float)
    b = np.array([[-1],[2],[-1]],float)
    epsilon = 1e-10 ; n = len(b)
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i]/a[i,i]
    dd = check(a,n)
    for j in range(n-(n-1)):
        if dd[j] == 1 and dd[j+1] == 1 and dd[j+2] == 1:
            print("A is a diagonally dominant matrix.")
        else:
            print("A is not a diagonally dominant matrix.")
    sol = gauss_seidel(a,b,epsilon,n,x)
    print("\n----------------------------------------------")
    sol_scipy = solve(a,b)
    print("\nSolution of the system of equations using scipy library:\n",sol_scipy)
    print("\n----------------------------------------------")
    table(sol)
    graph(sol,n)
    