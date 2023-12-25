import numpy as np
from scipy.linalg import solve

"Function checking for diagonal dominance of coefficient A matrix"
def diagonal_dominance(a,n):
    check = []
    for i in range(0,n):
        ele_sum = 0
        for j in range(0,n):
            if i != j:
                ele_sum = abs(ele_sum + a[i,j])
        if abs(a[i,i]) >= ele_sum:
            check.append(1)
        else:
            check.append(0)
    return check

"Function for Gauss Seidel method"
def Gauss_Seidel(a, b, epsilon, n, x):
    x1 = [x[0]]
    for i in range(N):
        for j in range(0, n):
            value = 0
            for k in range(0, n):
                if j != k:
                    value = value - a[j, k] * x[k]
            x[j] = 1 / a[j, j] * (b[j] + value)
            if j == 0:
                x1.append(x[j])
        if N > 1 and abs(x1[-1] - x1[-2]) <= epsilon:
            break
    print('\n Iterative Solution (using Gauss Seidel): \n', np.reshape(x, (n, 1)))

if __name__ == "__main__":
    "Example 1"
    #a = np.array([[1,-0.25,-0.25,0],[-0.25,1,0,-0.25],[-0.25,0,1,-0.25],[0,-0.25,-0.25,1]],float)
    #b = np.array([[50],[50],[25],[25]],float)
    "Exercise question"
    #a = np.array([[3,2,1],[1,3,2],[2,1,3]],float)
    #b = np.array([[7],[4],[7]],float)
    "3(a) Application: Electric Circuit"
    a = np.array([[4, -1, 0], [-1, 5, -1], [0, -1, 3]], float)
    b = np.array([[-1], [2], [-1]], float)
    "Divergence Matrix"
    #a = np.array([[2,3,5],[2,3,1],[3,0,9]])
    #b = np.array([[-1],[2],[3]])
    epsilon = 1e-16
    N = 100
    n = len(b)
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i]/a[i,i]
        dd = diagonal_dominance(a,n)
    for j in range(n-(n-1)):
        if dd[j] == 1 and dd[j+1] == 1 and dd[j+2] == 1:
            print("A is a diagonally dominant matrix.")
        else:
            print("A is not a diagonally dominant matrix.")
    Gauss_Seidel(a, b, epsilon, n, x)

    "Analytical Comparison with scipy linalg"
    Linalg = solve(a, b)
    print('\n Analytical Solution (using scipy linalg): \n', Linalg)
