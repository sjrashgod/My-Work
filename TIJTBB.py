import matplotlib.pyplot as plt
import pandas as pa
import numpy as np
from sklearn.linear_model import LinearRegression as lr


xs=pa.read_csv('C:\\Users\\ASUS\\Downloads\\Ferguson_Method.csv')
print(xs)
#print(xs["x=l^2"])
x1 = np.array(xs["x=l^2"]).reshape((-1,1))
y1 = np.array(xs["y=T^2.l"])
model= lr().fit(x1,y1)
y_cal = model.predict(x1)

plt.plot(x1,y_cal,c='blue')
plt.scatter(x1,y1,c='red')
plt.show()

#print(y_cal)

#print(x1)
#print(y1)