from MyIVP import elr, rk2, rk4
import numpy as np

def Func(yin, xval):
    y1, y2, y3 = yin
    f1 = y2 - y3 + xval
    f2 = 3*xval**2
    f3 = y2 + np.exp(-xval)
    eqn = [f1, f2, f3]
    return eqn

def elr_tol(Func, initial_cond, a, b, N, N_max, tol):
    w = 0
    Val = []
    N_arr = []
    count = 0
    while N <= N_max:
        g = elr(Func, initial_cond, a, b, N)
        # print("G:", g, "\n\n\n")
        t = np.linspace(a, b, N)
        Val.append(g)
        N_arr.append(N)
        if count >= 1:
            J = []
            K = []
            for i in range(len(initial_cond)):
                J.append(Val[-1][i][-1])
                K.append(Val[-2][i][-1])
            J=np.array(J)
            K=np.array(K)
            ff=[]
            for g1,g2 in zip(J,K):
                if abs(g1) <= 0.1e-5 or abs(g2) <= 0.1e-5:
                    err = abs(g1-g2)
                else:
                    err = abs((g2-g1)/g1)
                ff.append(err)
            if max(ff) <= tol:
                w = 1
                break
            else:
                pass
                  
        N = 2*N
        count += 1
    if w == 0:
        s = ("N_max reached without achieving required tolerance")
    elif w == 1:
      s = "Given tolerance achieved with",N_arr[-1],"sub-intervals"
   
    return(Val, Val[-1], N_arr[-1], N_arr, s, g, t)

def rk2_tol(Func, initial_cond, a, b, N, N_max, tol):
    w = 0
    Val = []
    N_arr = []
    count = 0
    while N <= N_max:
        g = rk2(Func, initial_cond, a, b, N)
        t = np.linspace(a, b, N)
        Val.append(g)
        N_arr.append(N)
        if count >= 1:
            J = []
            K = []
            for i in range(len(initial_cond)):
                J.append(Val[-1][i][-1])
                K.append(Val[-2][i][-1])
            J = np.array(J)
            K = np.array(K)
            ff = []
            for g1, g2 in zip(J, K):
                if abs(g1) <= 0.1e-5 or abs(g2) <= 0.1e-5:
                    err = abs(g1-g2)
                else:
                    err = abs((g2-g1)/g1)
                ff.append(err)
            if max(ff) <= tol:
                w = 1
                break
            else:
                pass
                  
        N = 2*N
        count += 1
    if w == 0:
        s = ("N_max reached without achieving required tolerance")
    elif w == 1:
      s = "Given tolerance achieved with", N_arr[-1], "sub-intervals"
   
    return(Val, Val[-1], N_arr[-1], N_arr, s, g, t)

def rk4_tol(Func, initial_cond,a,b,N,N_max,tol):
    w = 0
    Val = []
    N_arr = []
    count = 0
    while N <= N_max:
        g = rk4(Func, initial_cond,a,b,N)
        t = np.linspace(a, b, N)
        Val.append(g)
        N_arr.append(N)
        if count >= 1:
            J = []
            K = []
            for i in range(len(initial_cond)):
                J.append(Val[-1][i][-1])
                K.append(Val[-2][i][-1])
            J = np.array(J)
            K = np.array(K)
            ff = []
            for g1, g2 in zip(J,K):
                if abs(g1) <= 0.1e-5 or abs(g2) <= 0.1e-5:
                    err = abs(g1-g2)
                else:
                    err = abs((g2-g1)/g1)
                ff.append(err)
            if max(ff) <= tol:
                w = 1
                break
            else:
                pass
                  
        N = 2*N
        count += 1
    if w == 0:
        s = ("N_max reached without achieving required tolerance")
    elif w == 1:
      s = "Given tolerance achieved with",N_arr[-1],"sub-intervals"
   
    return(Val, Val[-1], N_arr[-1], N_arr, s, g, t)

if __name__ == '__main__':
    inc = [1, 1, -1]
    result = rk4_tol(Func, inc, 0, 1, 100, 1000, 10**-16)
    print(result)