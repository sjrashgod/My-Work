import matplotlib.pyplot as plt

x1 = [1,2,3]
y1 = [7,8,9]

x2 = [2,5,6]
y2 = [3,8,5]

plt.plot(x1,y1, label = 'first line')
plt.plot(x2,y2, label = 'second line')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title('matplotlib practice')
plt.legend()
plt.show()