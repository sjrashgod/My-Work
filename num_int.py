from scipy.integrate import quad

def integrand(x,s,t):
	return s / (t + (x**2))

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
		y = 1/(1 + (i**2))
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
		e1 = I[0] - simpson
		list_3.append(e1)
	else:
		e1 = None
		list_3.append(e1)
	e2 = I[0] - trapezoidal
	list_3.append(e2)
	return list_3 
	
if __name__ == "__main__":
	n = int(input("Enter the no. of intervals: "))
	b = float(input("Enter the uppper limit: "))
	a = float(input("Enter the lower limit: "))
	s = 1
	t = 1
	h = step_size(a,b,n)
	list_1 = interval_cal(a,h,n)
	list_2 = func(list_1)
	simpson = simpson_method(list_2,h,n)
	trapezoidal = trapezoidal_method(list_2,h)
	I = quad(integrand, 0, 1, args=(s,t))
	e = error(simpson,trapezoidal,I,n)
	print(I)
	#print(list_1)
	#print(list_2)
	print(simpson)
	print(trapezoidal)
	print(e)
