import numpy as np
import matplotlib.pyplot as plt


# PARTITION FUNCTION


def Z(g, e, T,k):
    z = []  # will store Z values for a particular energy for all tempratures
    p=0
    for i in e:
        z_i = g[p] * np.e**(-i / (k * T))
        z.append(z_i)
        p+=1
    return np.array(z)


g = [1,1,1]
k = 8.617 * 10 ** (-5)
e = [0, 1,2]

T = np.linspace(0.001, 100000, 500)
tmin=0
for i in T:
    if i<=(e[-1]/k):
        tmin+=1
#print(np.sum(Z(g, e, T,k)))

y=np.zeros(len(T))
for i in range(len(e)):
    y += Z(g, e, T,k)[i]

plt.plot(T[:tmin],y[:tmin], c='blue')
plt.plot(T[tmin:],y[tmin:], c='red')
plt.grid()
plt.xlabel('T')
plt.ylabel('Z')
plt.title('Z VS T for all temperatures')
plt.show()

plt.scatter(T[:tmin],y[:tmin], c='blue')
plt.grid()
plt.xlabel('T')
plt.ylabel('Z')
plt.title('Z VS T for low temperatures')
plt.show()

plt.scatter(T[tmin:],y[tmin:], c='red')
plt.grid()
plt.xlabel('T')
plt.ylabel('Z')
plt.title('Z VS T for high temperatures')
plt.show()


#FRACTIONAL PROBABILITY

Y2=[]
p=0
for j in e:
    Y2.append(g[p]*np.e**(-j/(k*T)) / y)
    p+=1

for i in range(len(e)):
    plt.scatter(T,Y2[i])
plt.plot(T,np.ones(len(T)) * 1/len(e),linestyle='dashdot')
plt.grid()
plt.title('Ni/N VS T for all temperatures')
plt.xlabel('T')
plt.ylabel('Fractional Probability')
plt.show()

#plt.show()


#INTERNAL ENERGY
y3=np.zeros(len(T))
p=0
for i in e:
    y3+= (g[p]*np.e**(-np.array(i)/(k*T)))*i / y
    p+=1
plt.plot(T[:tmin],y3[:tmin], c='red')
plt.plot(T[tmin:],y3[tmin:], c='blue')
plt.title('U/N VS T for all temperatures')
plt.xlabel('T')
plt.ylabel('U/N')
plt.show()

plt.scatter(T[:tmin],y3[:tmin])
plt.title('U/N VS T for low temperatures')
plt.grid()
plt.show()
plt.scatter(T[tmin:],y3[tmin:])
plt.title('U/N VS T for high temperatures')
plt.xlabel('T')
plt.ylabel('U/N')
plt.grid()
plt.show()

# ENTROPY
U=y3
N = 1
S= (N*k*np.log(y/N))+ U/(N*T) + N*k
plt.plot(T[:tmin],S[:tmin])
plt.plot(T[tmin:],S[tmin:])
plt.title('Entropy VS T')
plt.xlabel('T')
plt.ylabel('Entropy')
plt.grid()
plt.show()

S2= N*k*np.log(y[tmin:]/N) + U[tmin:]/(N*T[tmin:]) + N*k
plt.scatter(T[tmin:],S2)
plt.grid()
plt.title('Entropy VS T for high Temperature')
plt.xlabel('T')
plt.ylabel('Entropy')
plt.show()

# FREE ENERGY

F=-N*k*T*np.log(y)
plt.plot(T[:tmin],F[:tmin])
plt.plot(T[tmin:],F[tmin:])
plt.grid()
plt.title('Free Energy VS T for all Temperature')
plt.xlabel('T')
plt.ylabel('Free Energy')
plt.show()

plt.scatter(T[:tmin],F[:tmin])
plt.grid()
plt.title('Free Energy VS T for low Temperature')
plt.xlabel('T')
plt.ylabel('Free Energy')
plt.show()

plt.scatter(T[tmin:],F[tmin:])
plt.title('Free Energy VS T for high Temperature')
plt.xlabel('T')
plt.ylabel('Free Energy')
plt.grid()
plt.show()

