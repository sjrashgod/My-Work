import matplotlib.pyplot as plt
import pandas as pa
import numpy as np

xs=pa.read_csv("C:/Users/Sarthak/Desktop/Python/Python Work/HW/BP.csv")
print(xs)

l=np.array(xs['L'])
t=np.array(xs['T'])
t2=np.array(xs['T2'])

plt.plot(l,t, -l,t2)
plt.xlabel('Distance from CG (L)', fontsize=14)
plt.ylabel('Time Period (T)', fontsize=14)
plt.title('Bar Pendulum', fontsize=16)
plt.show()