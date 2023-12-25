import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.linear_model import LinearRegression
import statistics

def planck_1(x):
    return x**3/(np.exp(x)-1)

x=np.linspace(0.001,12,5000)
y=planck_1(x)
y_max=np.max(y)

p=0
for i in y:
    if i==y_max:
        break
    else:
        p+=1

x_max=x[p]
h = 6.67 * (10 ** (-34))
c = 3* (10**(8))
k = 1.380649 * (10 ** (-23))
b = (h*c)/(k*x_max)
print(b)


plt.scatter(x,planck_1(x))
plt.scatter(x_max,y_max,c='r')
plt.title("Density Of States For Reyleigh-Jeans Criterian")
plt.xlabel('x')
plt.ylabel('G(x)')
plt.grid()
plt.show()

def MySimp(a, b, n, f,p):
    X = []
    I=[]
    tol=0.5*10**(-5)
    count1=0
    count2=0
    for i in range(1,b):
        count1+=1
        x = np.linspace(a, i*10, 10*(n + 1))
        X.append(x)
        h = (i*10 - a) / (10*n)

        y = f(x)/(np.e**(p(x)) - 1)

        integral = (h / 3) * (2 * np.sum(y[2:-2:2]) + 4 * np.sum(y[1:-1:2]) + y[0] + y[-1])
        I.append(integral)
        if count1!=1:
            if abs((I[-1]-I[-2])/I[-2]) < tol:
                count2+=1
                if count2>100:
                    #print(X[-1])     #Value Of b till where Integration is done
                    #print(count1)    #Total Iterations after which desired tolerance is achieved
                    #T=x
                    break

    #x_i=np.linspace(0.01,X[-1],500)

    return I[-1],X[-1]

def half_a(b):
    x2=np.linspace(0.001,b,500)

    y = (x2**3 / (np.e ** (x2) - 1))
    h=abs(x2[1]-x2[0])
    integral = (h / 3) * (2 * np.sum(y[2:-2:2]) + 4 * np.sum(y[1:-1:2]) + y[0] + y[-1])
    print(integral)
    h = 6.67 * (10 ** (-34))
    return integral*((8*np.pi* (k*6000)**4)/((h*c)**3)),x2


val1=[]

print("Function")
func = eval("lambda x:" + input("Function :"))
power = eval("lambda x:" + input("Exponent of e :"))

f = lambda x: func(x)
p = lambda x: power(x)

val1.append(MySimp(0.001,200,1000,f,p)[0])

print(val1)

I=val1[0]*((8*np.pi* (k*6000)**4)/((h*c)**3))

val2=MySimp(0.001,200,1000,f,p)[1]
p1=0


for i in val2[1:]:
    if half_a(i)[0] <= (I/2):
        print(half_a(i)[0]-I/2)
        p1+=1
    else:
        print("index",p1)
        #print(abs(half_a(i)[0] - (I / 2)))
        break

#print(len(val2))


T=[]
for i in range(100,10500,500):
    T.append(i)
T=np.array(T)
u=((8*np.pi* (k*T)**4)/((h*c)**3))* val1[0]
F=u* (c/4)
#print(u)

plt.plot(T,F)
plt.xlabel('T')
plt.ylabel('Flux')
plt.grid()
plt.show()

T_new=np.log(T)
F_new=np.log(F)

plt.plot(T_new,F_new)
plt.xlabel('log T')
plt.ylabel('log Flux')
plt.grid()
plt.show()

T_new=T_new.reshape(-1,1)
model=LinearRegression().fit(T_new,F_new)
#r_sq=model.score(x,y)
print("slope: ", model.coef_)
print("intercept: ", model.intercept_)