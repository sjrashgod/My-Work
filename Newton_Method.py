import math
def func(g):
	return((g-8)*math.tan(g))
def deriv(g):
	return((math.tan(g)+(g-8)*(1/math.cos(g))**2))
def new_num(f):
	return(f-(func(f)/deriv(f)))


def newton_method(x):
    	x_new = new_num(x)
    	print('\n------------------------------------------------')
    	print('| terminating number          | relative error |')
    	print('------------------------------------------------')
    	while abs(x_new - x)>0.000000001:
    		x,x_new = new_num(x),new_num(x_new)
    		print('| ',x_new,' '*(25-len(str(x_new))),'|')
    		
    	print('------------------------------------------------\n')
    	return(x_new) 
newton_method(8.5)

