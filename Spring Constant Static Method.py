import matplotlib.pyplot as plt
import pandas as pa
import numpy as np
from sklearn.linear_model import LinearRegression as lr

xs=pa.read_csv("C:/Users/Sarthak/Desktop/Python/Python Work/HW/StaticSet1.csv")
print(xs)
M=np.array(xs["M = x"]).reshape((-1,1))
li=np.array(xs["load inc"])
ld=np.array(xs["load dec"])
L=((li+ld)/2)
model= lr().fit(M,L)
y_cal = model.predict(M)
print("y_cal=",y_cal )
plt.plot(M,y_cal,c='violet')
plt.scatter(M,L,c='blue',alpha=0.7,edgecolors='black')
plt.xlabel('Load M', fontsize=14)
plt.ylabel('Elongation L', fontsize=14)
plt.title('Spring Constant Static Set 1', fontsize=16)
plt.show()

xs=pa.read_csv("C:/Users/Sarthak/Desktop/Python/Python Work/HW/StaticSet2.csv")
print(xs)
M=np.array(xs["M = x"]).reshape((-1,1))
li=np.array(xs["load inc"])
ld=np.array(xs["load dec"])
L=((li+ld)/2)
model= lr().fit(M,L)
y_cal = model.predict(M)
print("y_cal=",y_cal )
plt.plot(M,y_cal,c='green')
plt.scatter(M,L,c='red',alpha=0.7,edgecolors='black')
plt.xlabel('Load M', fontsize=14)
plt.ylabel('Elongation L', fontsize=14)
plt.title('Spring Constant Static Set 2', fontsize=16)
plt.show()
