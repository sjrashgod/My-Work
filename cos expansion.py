import math
import numpy as np
import matplotlib.pyplot as plt 
def func_cos(x,n):
    cos_approx = 0
    for i in range(n):
        cos_approx += (-1)**i * x**(2*i)/ math.factorial(2*i)
    return cos_approx

angls=np.arange(-2*np.pi,2*np.pi,0.1)
cos_angls = np.cos(angls)

fig, ax = plt.subplots()
ax.plot(angls,cos_angls)
for j in range (1,10):
    apr_cos = [func_cos(ang,j) for ang in angls]
    plt.plot(angls,apr_cos)
    
ax.set_ylim([-5,5])
plt.show()    
    
