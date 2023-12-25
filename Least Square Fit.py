def lsqf(x, y,):
    n1=len(x)
    n2=len(y)
    a1 = 0
    a0 = 0
    if n1 == n2 and n1 > 3:
       sigma_xi = 0
       sigma_yi = 0
       sigma_xi_yi = 0
       sigma_xisq = 0

       count = 0
       while count <n1:
           sigma_xi = sigma_xi + x[count]
           sigma_yi = sigma_yi + y[count]
           sigma_xi_yi = sigma_xi_yi + x[count] * y[count]
           sigma_xisq = sigma_xisq + x[count]**2
           count = count + 1

    a1 = (n1 * sigma_xi_yi - sigma_xi * sigma_yi) / (n1 * sigma_xisq - sigma_xi**2)
    a0 = (sigma_xisq * sigma_yi - sigma_xi * sigma_xi_yi) / (n1 * sigma_xisq - sigma_xi**2)
    
    return a1,a0

xx=[750,1000,1250,1500,1750,2000,2250]

yy=[33.75,39,44.02,48.76,55.59,60.91,65.77]

m,c = lsqf(xx,yy)
print(m,c)




