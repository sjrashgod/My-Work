import numpy as np
import matplotlib.pyplot as plt
import MyIntegration as mi
import pandas as pd

"Name: Sarthak Jain, 2020PHY1201"
"Partner Name: Ishmeet Singh, 2020PHY1221"

print("My Roll No.: 2020PHY1201")

def integral_simp(I1_exact,I2_exact):
    i = 2
    I1_simp = []
    I2_simp = []
    count1_simp = []
    count2_simp = []
    I1_exact_S = []
    I2_exact_S = []

    while True:
        if abs(I1_exact - mi.MySimp("exp(-x)/(1+x**2)",1000,0,i)) <= 10**(-2):
            I1_simp.append(mi.MySimp("exp(-x)/(1+x**2)",1000,0,i))
            I1_exact_S.append(I1_exact)
            count1_simp.append(i)
            break
        else:
            I1_simp.append(mi.MySimp("exp(-x)/(1+x**2)",1000,0,i))
            I1_exact_S.append(I1_exact)
            count1_simp.append(i)
            i += 2

    i = 2

    while True:
        if abs(I2_exact - mi.MySimp("1/(1+x**2)",1000,0,i)) <= 10**(-2):
            I2_simp.append(mi.MySimp("1/(1+x**2)",1000,0,i))
            I2_exact_S.append(I2_exact)
            count2_simp.append(i)
            break
        else:
            I2_simp.append(mi.MySimp("1/(1+x**2)",1000,0,i))
            I2_exact_S.append(I2_exact)
            count2_simp.append(i)
            i += 2

    return I1_simp,I2_simp,count1_simp,count2_simp,I1_exact_S,I2_exact_S

def graph(I1,I2,n,I1_simp,I2_simp,count1_simp,count2_simp,I1_exact_S,I2_exact_S):
    fig1,ax1 = plt.subplots(1, 2)
    fig2,ax2 = plt.subplots(1, 2)
    ax1[0].plot(n,I1,label = "MyLagQuad")
    ax1[0].plot(n,I1_exact_LL,label = "Analytic Value")
    ax1[1].plot(count1_simp,I1_simp,label = "MySimp")
    ax1[1].plot(count1_simp,I1_exact_S,label = "Analytic Value")
    ax2[0].plot(n,I2,label = "MyLagQuad")
    ax2[0].plot(n,I2_exact_LL,label = "Analytic Value")
    ax2[1].plot(count2_simp,I2_simp,label = "MySimp")
    ax2[1].plot(count2_simp,I2_exact_S,label = "Analytic Value")
    for i in range(2):
        if i == 0:
            ax1[i].set(xlabel = "Nodal Points (n)",ylabel = "Value of Integration (I)",title = "Gauss Laguerre Quadrature")
            ax2[i].set(xlabel = "Nodal Points (n)",ylabel = "Value of Integration (I)",title = "Gauss Laguerre Quadrature")
        elif i == 1:
            ax1[i].set(xlabel = "Nodal Points (n)",ylabel = "Value of Integration (I)",title = "Simpson 1/3 Method")
            ax2[i].set(xlabel = "Nodal Points (n)",ylabel = "Value of Integration (I)",title = "Simpson 1/3 Method")
        ax1[i].grid(ls = "--")
        ax2[i].grid(ls = "--")
        ax1[i].legend()
        ax2[i].legend()
    fig1.suptitle("INTEGRAL 1")
    fig2.suptitle("INTEGRAL 2")
    plt.show()
        
if __name__ == "__main__":
    
    # PART B I

    count = 0
    func = []
    Exact=[1,1,2,6,24,120,720,5040,40320,362880]
    
    for count in range(len(Exact)):
        f = input("\nEnter Function: ")
        func.append(f)
        if count < (len(Exact) - 1):
            ans = input("Do you want to enter more function (Y/N) ?\t")
            if ans == "N" or ans == "n":
                break

    for j,m in zip(func,Exact):
        for k in range(2,6,2):
            print("\nValue of integration of",j,"for n =",k,"is: ",mi.MyLaguQuad(j,k))
        print("\nExact Value of integration of",j,"is: ",m)
        print("---------------------------------------------------------------")

    
    # PART B II

    I1 = []
    I2 = []
    I1_exact = 0.621449624235813
    I2_exact = 1.570796326794897
    I1_exact_LL = []
    I2_exact_LL = []
    n = []
    
    for i in range(2,130,2):
        n.append(i)
        I1_exact_LL.append(I1_exact)
        I2_exact_LL.append(I2_exact)
        i1 = mi.MyLaguQuad("1/(1+x**2)",i)
        I1.append(i1)
        i2 = mi.MyLaguQuad("exp(x)/(1+x**2)",i)
        I2.append(i2)

    data1 =  np.column_stack([n,I1,I2])
    file1 = np.savetxt("quad-lag-1221.txt",data1,header = ("n,I1,I2"))

    df1 =  pd.DataFrame({"n": n, "I1": I1, "I2": I2})
    print("\nGAUSS LAGUERRE QUADRATURE:\n",df1)

    # PART B III & IV

    I1_simp,I2_simp,count1_simp,count2_simp,I1_exact_S,I2_exact_S = integral_simp(I1_exact,I2_exact)

    df2 =  pd.DataFrame({"n": count1_simp, "I1": I1_simp})
    print("\nTOLERNACE LIMIT = 10**(-2)")
    print("\nSIMPSON FOR INTEGRAL 1:\n",df2)
    data2 =  np.column_stack([count1_simp,I1_simp])
    file2 = np.savetxt("Simpson-Integral_1-1221.txt",data2,header = ("n,I1"))

    print("\n-----------------------------------------------------------------")

    df3 =  pd.DataFrame({"n": count2_simp, "I1": I2_simp})
    print("\nTOLERNACE LIMIT = 10**(-2)")
    print("\nSIMPSON FOR INTEGRAL 1:\n",df3)
    data3 =  np.column_stack([count2_simp,I2_simp])
    file3 = np.savetxt("Simpson-Integral_2-1221.txt",data3,header = ("n,I2"))

    graph(I1,I2,n,I1_simp,I2_simp,count1_simp,count2_simp,I1_exact_S,I2_exact_S)

