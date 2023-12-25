import MyIntegration as mi 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from tabulate import tabulate

#C
def mypi_trap(a,b,n):
	inte_trap = np.array([mi.MyTrap("4/(1 + x**2)",b,a,i) for i in n])
	return inte_trap

def mypi_simp(a,b,n):
	inte_simp = np.array([mi.MySimp("4/(1 + x**2)",b,a,i) for i in n])
	return inte_simp
def error(pi,inte_simp,inte_trap):
	err_t = abs(inte_trap - pi)
	err_s = abs(inte_simp - pi)
	return [err_t,err_s]
#D
def table(a,b):
	d = int(input("\nEnter the number of Significant digits: "))
	dict = {'Methods': ["Trapezoidal Method", "Simpson 1/3 Method"],
            'my_pi(n)':[mi.MyTrap_tol("4/(1 + x**2)",b,a,1000,d)[0],mi.MySimp_tol("4/(1 + x**2)",b,a,1000,d)[0]],
            'n': [mi.MyTrap_tol("4/(1 + x**2)",b,a,1000,d)[1],mi.MySimp_tol("4/(1 + x**2)",b,a,1000,d)[1]],
            'E = |my_pi(n) - pi|/pi': [abs(mi.MyTrap_tol("4/(1 + x**2)",b,a,1000,d)[0] - np.pi)/np.pi,abs(mi.MySimp_tol("4/(1 + x**2)",b,a,1000,d)[0] - np.pi)/np.pi]
            }
	df = pd.DataFrame(dict)
	print("\n",df, "\n") 
	
def graph(n,inte_simp,inte_trap,pi,h,err):
	fig1,ax1 =  plt.subplots()
	fig2,ax2 = plt.subplots()
	fig3,ax3 = plt.subplots()
	fig4,ax4 = plt.subplots()
	fig4,ax5 = plt.subplots()
	fig4,ax6 = plt.subplots()
	ax1.plot(n,inte_trap,label = "Approximate $\pi$ (Trapezoidal)")
	ax1.scatter(n,inte_trap,s = 25)
	ax1.plot(n,pi,label = "Original $\pi$")
	ax2.plot(n,inte_simp,label = "Approximate $\pi$ (Simpson)")
	ax2.scatter(n,inte_simp,s = 25)
	ax2.plot(n,pi,label = "Original $\pi$")
	ax3.plot(n,err[0])
	ax4.plot(n,err[1])
	ax5.plot(h,err[0])
	ax6.plot(h,err[1])
	ax3.scatter(n,err[0],s = 25)
	ax4.scatter(n,err[1],s = 25) 
	ax1.set(xlabel = "No. of intervals (n)",ylabel  = "Value of $\pi$",title = "Value of $\pi$ Vs No. of intervals")
	ax2.set(xlabel = "No. of intervals (n)",ylabel  = "Value of $\pi$",title = "Value of $\pi$ Vs No. of intervals")
	ax3.set(xlabel = "No of intervals (n)",ylabel = "e = |my_pi - $\pi$|",title = "Error Vs No. of intervals (Trapezoidal)")
	ax4.set(xlabel = "No of intervals (n)",ylabel = "e = |my_pi - $\pi$|",title = "Error Vs No. of intervals (Simpson)")
	ax5.set(xlabel = "Log of step size ln(h)",ylabel = "Log of error ln(e)",title = "ln(e) Vs ln(h) (Trapezoidal)")
	ax6.set(xlabel = "Log of step size ln(h)",ylabel = "Log of error ln(e)",title = "ln(e) Vs ln(h) (Simpson)")
	ax1.grid(ls = "--")
	ax2.grid(ls = "--")
	ax3.grid(ls = "--")
	ax4.grid(ls = "--")
	ax5.grid(ls = "--")
	ax6.grid(ls = "--")
	ax6.set_yscale("log")
	ax1.legend()
	ax2.legend()
	ax5.set_xscale("log")
	ax5.set_yscale("log")
	ax6.set_xscale("log")
	plt.show()

#E 1
def pi_quad(a,b,f):
	Order = [2,4,8,16,32,64]
	SubInt =[1,2,4,8,16,32] #subintevals

	pi_quad = np.zeros((len(Order),len(SubInt)))
	for i in range(len(Order)):
		for j in range(len(SubInt)):
			pi_quad[i,j]= mi.MyLegQuadrature(a,b,Order[i],f,SubInt[j])
	return pi_quad

#E 2
def plotting(pi_array):
	Order = [2,4,8,16,32,64]
	y1 = []
	y2 = []
	for i in pi_array:
		y1.append(i[0])
	for j in pi_array:
		y2.append(j[2])
	plt.scatter(Order,y1)
	plt.plot(Order,y1,label= 'm = 1')
	plt.scatter(Order,y2)
	plt.plot(Order,y2,label= 'm = 8')
	plt.xlabel("nth-point Quadrature")
	plt.ylabel("$\pi$ value")
	plt.title("$\pi$ vs nth order gauss leg quadrature")
	plt.legend()
	plt.grid(True,ls='--')
	plt.show()
	
	
#F part 
def pi_tol(a,b,f,m_max):
	Order = [2,4,8,16,32,64]
	tol = [1,2,3,4,5,6,7,8]
	pi_tol = np.zeros((len(Order),len(tol)),dtype=list)
	for i in range(len(Order)):
		for j in range(len(tol)):
			pi_tol[i,j] = np.array(mi.MyLegQuadrature_tol(a,b,f,Order[i],tol[j],m_max))
	return pi_tol




if __name__ == "__main__":
	a = 0 ; b = 1 # LOWER AND UPPER LIMIT
	n = np.array([2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32])
	h = np.array([(b-a)/j for j in n])
	pi = np.array([np.pi for i in range(len(n))])

	inte_trap = mypi_trap(a,b,n)
	inte_simp = mypi_simp(a,b,n)
	err = error(pi,inte_simp,inte_trap)
	    
	table(a,b)
	graph(n,inte_simp,inte_trap,pi,h,err)

	
	nn= ["n=2","n=4","n=8","n=16","n=32","n=64"]
	pi_array = pi_quad(0,1,"4/(1+x**2)")
	df1= pd.DataFrame(np.column_stack((nn,pi_array)))
	print("Values of π approximated by Gauss Legendre Quadrature with n−point formula with m subintervals.")
	print(tabulate(df1,headers=["","m=1","m=2","m=4","m=8","m=16","m=32"],showindex=False,tablefmt="grid"),"\n")
	r = open("pi_quad2020PHY1201.dat","w")#saving the data 
	content = str(np.column_stack((nn,pi_array)))
	r.write(content)
	r.close()
	plotting(pi_array)
	
	res = abs(pi_quad(0,1,"4/(1+x**2)")-np.pi)
	df2=pd.DataFrame(np.column_stack((nn,res)))
	print("Table for error in pi calculation ")
	print(tabulate(df2,headers=["","m=1","m=2","m=4","m=8","m=16","m=32"],showindex=False,tablefmt="grid"),"\n")
	df= pd.DataFrame(np.column_stack((nn,(pi_tol(0,1,"4/(1+x**2)",200)))))
	print("alues of pi approximated by Gauss Legendre Quadrature with n−point formula with tolerance tol.")
	print(tabulate(df,headers=["","tol=0.5*10**-1","tol=0.5*10**-2","tol=0.5*10**-3","tol=0.5*10**-4","tol=0.5*10**-5","tol=0.5*10**-6","tol=0.5*10**-7","tol=0.5*10**-8"],showindex=False,tablefmt="grid"))
	
	
