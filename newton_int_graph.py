from scipy.integrate import quad
import matplotlib.pyplot as plt

def integrand(x):
	return (x**2)

def step_size(a,b,n):
	h = (b - a)/n
	return h
	
def interval_cal(a,h,n):
	list_1 = []
	list_1.append(a)
	for i in range(n):
		a = a + h
		list_1.append(a)				
	return list_1
	
def func(list_1):
	list_2 = []
	for i in (list_1):
		y = (i**2)
		list_2.append(y)
	return list_2
	
def simpson_method(list_2,h,n):
	z2,z3,inte = 0,0,0
	if n%2 == 0:
		z1 = list_2[0] + list_2[len(list_2)-1]
		for i in range(1,len(list_2)-1):
			if i%2 == 0:
				z2 = z2 + list_2[i]
			else:
				z3 = z3 + list_2[i]
		#print(z1,z2,z3)
		inte = h/3 * (z1 + (4 * (z3)) + (2 * (z2)))
	else:
		print("Simpson Rule is not applicable since interval is not even.")
		inte = None
	return inte
	
def trapezoidal_method(list_2,h):
	z2 = 0
	z1 = (list_2[0] + list_2[len(list_2)-1])/2
	for i in range(1,len(list_2)-1):
		z2 = z2 + list_2[i]
	#print(z2)
	inte = h * (z1 + z2)
	return inte
	
def error(simpson,trapezoidal,I,n):
	list_3 = []
	if n%2 == 0:
		e1 = abs(I[0] - simpson)
		list_3.append(e1)
	else:
		e1 = None
		list_3.append(e1)
	e2 = abs(I[0] - trapezoidal)
	list_3.append(e2)
	return list_3 
	
if __name__ == "__main__":
	
	
	ans = input("Single Valued Function or Mutiple Valued Function (S/M) ?")
	
	if(ans == "S" or ans == "s"):
		n = int(input("Enter the no. of intervals: "))
		b = float(input("Enter the uppper limit: "))
		a = float(input("Enter the lower limit: "))
		h = step_size(a,b,n)
		list_1 = interval_cal(a,h,n)
		list_2 = func(list_1)
		simpson = simpson_method(list_2,h,n)
		trapezoidal = trapezoidal_method(list_2,h)
		I = quad(integrand, 0, 1)
		e = error(simpson,trapezoidal,I,n)
		print('Value of stepsize is:',h)
		print('Value of integral using scipy',I[0])
		#print(list_1)
		print('Value of function at every nodal point:',list_2)
		print('Value of integral using simpson method:',simpson)
		print('Value of integral using trapezoidal method:',trapezoidal)
		print('Error in both method (simpson,trapezoidal)',e)	
	
	
	#s = 1
	#t = 1
	if(ans == "M" or ans == "m"):
		list_h = []
		list_simpson = []	
		list_trapezoidal = []
		for n in range(2,101):
			a = 0  #lower limit
			b = 1  # upper limit
			h = step_size(a,b,n)
			list_1 = interval_cal(a,h,n)
			list_2 = func(list_1)
			simpson = simpson_method(list_2,h,n)
			trapezoidal = trapezoidal_method(list_2,h)
			I = quad(integrand, 0, 1)
			e = error(simpson,trapezoidal,I,n)
			print('Value of stepsize is:',h)
			print('Value of integral using scipy',I[0])
			#print(list_1)
			#print('Value of function at every nodal point:',list_2)
			print('Value of integral using simpson method:',simpson)
			print('Value of integral using trapezoidal method:',trapezoidal)
			print('Error in both method (simpson,trapezoidal)',e)
			list_h.append(h)
			list_simpson.append(simpson)
			list_trapezoidal.append(trapezoidal)
			print("------------------------------------")
		print(len(list_h))
		print(len(list_simpson))
		print(len(list_trapezoidal))

		plt.scatter(list_h,list_simpson,color = 'r')
		plt.scatter(list_h,list_trapezoidal,color = 'b')
		plt.show()
