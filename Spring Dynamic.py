import matplotlib.pyplot as plt
import pandas as pa
import numpy as np
from sklearn.linear_model import LinearRegression as lr

xs=pa.read_csv("C:/Users/Sarthak/Desktop/Python/Python Work/HW/Dynamic.csv")
print(xs)
M=np.array(xs["M = x"]).reshape((-1,1))
T=np.array(xs["T"])
y=T**2
model= lr().fit(M,y)
y_cal = model.predict(M)
print("y_cal=",y_cal )
plt.plot(M,y_cal,c='purple')
plt.scatter(M,y,c='yellow',s=50, alpha=1,edgecolors='black')
plt.xlabel('Load M', fontsize=14)
plt.ylabel('$T^2$', fontsize=14)
plt.title('Spring Constant Dynamic Method', fontsize=16)
plt.show()
