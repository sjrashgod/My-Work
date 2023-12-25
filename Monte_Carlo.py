import random as rd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


count=0
for n in range(10**4):
	x = np.random.random()
	y = np.random.random()
	if (x-0.5)**2 + (y-0.5)**2 <= 1/4:
		count+=1
		
#print(count)
print((count/10**4)*4)

x= np.arange(0,1,0.01)
avg=0
area=0.0
for a in x :
	avg+=np.sqrt(1-a**2)
	area+=np.sqrt(1-a**2)*0.01
avg=avg/100
print(area*4)




