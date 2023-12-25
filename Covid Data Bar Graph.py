import matplotlib.pyplot as plt
import pandas as pa
import numpy as np

xs=pa.read_csv("C:/Users/Sarthak/Desktop/Python/Python Work/HW/CovidData.csv")
x=np.array(xs["month"])
a=np.array(xs["confirmed cases"])
b=np.array(xs["deaths"])
c=np.array(xs["recoveries"])
y=[a,b,c]
x=np.arange(11)
b1=plt.bar(x-0.25,y[0],color='red', width=0.25,)
b2=plt.bar(x+0.00,y[1],color='green',width=0.25)
b3=plt.bar(x+0.25,y[2],color='blue',width=0.25)
labels=['Confirmed cases','Deaths','Recoveries']
plt.xticks(x,["Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan"])
plt.xlabel('Months', fontsize=14)
plt.ylabel('No. of people', fontsize=14)
plt.legend(labels)
plt.title('Covid Data Bar Graph for Haryana', fontsize=16)
plt.show()

          
           