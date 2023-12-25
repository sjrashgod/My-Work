import matplotlib.pyplot as plt
import pandas as pa
import numpy as np
from sklearn.linear_model import LinearRegression as lr

xs=pa.read_csv("C:/Users/Sarthak/Desktop/Python/Python Work/HW/FergusonSet1.csv")
print(xs)
x=np.array(xs["x=l^2"]).reshape((-1,1))
y=np.array(xs["y=T^2.l"])
model= lr().fit(x,y)
y_cal = model.predict(x)
print("y_cal=",y_cal )
plt.plot(x,y_cal,c='blue')
plt.scatter(x,y,c='red',alpha=0.7,edgecolors='black')
plt.xlabel('x=L^2', fontsize=14)
plt.ylabel('y=T^2.l', fontsize=14)
plt.title('Ferguson Method Set 1', fontsize=16)
plt.show()

xs=pa.read_csv("C:/Users/Sarthak/Desktop/Python/Python Work/HW/FergusonSet2.csv")
print(xs)
x=np.array(xs["x=l^2"]).reshape((-1,1))
y=np.array(xs["y=T^2.l"])
model= lr().fit(x,y)
y_cal = model.predict(x)
print("y_cal=",y_cal )
plt.plot(x,y_cal,c='green')
plt.scatter(x,y,c='orange',alpha=0.7,edgecolors='black')
plt.xlabel('x=L^2', fontsize=14)
plt.ylabel('y=T^2.l', fontsize=14)
plt.title('Ferguson Method Set 2', fontsize=16)

plt.show()