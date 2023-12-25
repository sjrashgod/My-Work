import numpy as np
import scipy.linalg

def gauss_elimination(a,b,n):
    for k in range(n-1):
        if np.fabs(a[k,k]) == 0:
            for i in range(k+1,n):
                if np.fabs(a[i,k]) > np.fabs(a[k,k]):
                    a[[k,i]] = a[[i,k]]
                    b[[k,i]] = b[[i,k]]
                    print("\nMatrix A after row exchange:\n",a)
                    print("\nMatrix B after row exchange:\n",b)
                    print("\n-------------------------------------------------------")
                    break
        for i in range(k+1,n):
            if a[i,k] == 0:
                continue
            factor = a[k,k]/a[i,k]
            for j in range(k,n):
                a[i,j] = a[k,j] - a[i,j]*factor      
            b[i] = b[k] - b[i]*factor
            print("\nMatrix A after next step of elimination:\n",a) 
            print("\nMatrix B after next step of elimination:\n",b)
    return [a,b]

def rank(sol,n):
    a = sol[0] ; b = sol[1] ; rank_a = 0 ; rank_arg = 0
    arg_matrix = np.concatenate((a,b),axis = 1)
    for i in range(n):
        if a[i,i] != 0:
            rank_a += 1
    for j in range(np.shape(arg_matrix)[0]):
         for k in range(np.shape(arg_matrix)[1]):
             if arg_matrix[j,k] != 0:
                 rank_arg +=1
                 break
    print("\nMatrix A after elimination process:\n",sol[0])
    print("\nMatrix B after elimination process:\n",sol[1])
    print("\nAugmented Matrix:\n",arg_matrix)
    print("\nRank of cofficient matrix A is:",rank_a)
    print("\nRank of augmented matrix A|B is:",rank_arg)
    return [rank_a,rank_arg]
            
def back_substitution(sol,n,r):
    if r[0] == r[1] and r[0] == n:
        print("\nSystem of equations has a unique solution.")
        x = np.zeros(n)
        x[n-1] = sol[1][n-1]/sol[0][n-1,n-1]   # sol[0] = a & sol[1] = b
        for i in range(n-2,-1,-1):
            ax = 0
            for j in range(i+1,n):
                ax = ax + sol[0][i,j]*x[j]
            x[i] = (sol[1][i] - ax)/sol[0][i,i]
        print("\nSolution of the system of equations is:\n",np.reshape(x,(n,1)))
    elif r[0] < r[1]:
        print("\nSystem of equations has no solution.")
    elif r[0] == r[1] and r[0] < n:
        print("\nSystem of equations has infinetly many solution.")

if __name__ == "__main__":
    a = np.array([[4,-1,0],[-1,5,-1],[0,-1,3]],float)
    b = np.array([[-1],[2],[-1]],float)
    n = len(b)
    var = np.zeros(n)
    print("\nMatrix A:\n",a)
    print("\nMatrix B:\n",b)
    print("\n-----------------------------------------------------")
    sol = gauss_elimination(a,b,n)
    print("\n-----------------------------------------------------")
    r = rank(sol,n)
    print("\n-----------------------------------------------------")
    back = back_substitution(sol,n,r)
    print("\n-----------------------------------------------------")
    if r[0] == r[1] and r[0] == n:
        print("\nSolution using scipy library:")
        print("\n",scipy.linalg.solve(a,b))
