import numpy as np
from sklearn.linear_model import LinearRegression
x=np.array([5,15,25,35,45,55]).reshape((-1,1))
y=np.array([5,20,14,32,22,38])
print(x)
model = LinearRegression().fit(x,y)
r_sq = model.score(x,y)
print('coefficient of determinant:',r_sq)
print('slope:',model.coef_)

