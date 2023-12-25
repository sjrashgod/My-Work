'''Ans 1'''

import numpy as np
import cmath
psi_1=np.array([[1],[0]])
psi_2=np.array([[0],[1]])

print(psi_1,"\n",psi_2,"\n")

'''Ans 2'''

sigma_x=np.array([[0,1],[1,0]])
sigma_y=np.array([[0,complex(0,-1)],[complex(0,1),0]])
sigma_z=np.array([[1,0],[0,-1]])
sigma_refer={1:sigma_x,2:sigma_y,3:sigma_z}


print(sigma_x,"\n\n",sigma_y,"\n\n",sigma_z,"\n\n")

'''Ans 3'''

sigma2_x = sigma_x.dot(sigma_x)
sigma2_y = sigma_y.dot(sigma_y)
sigma2_z = sigma_z.dot(sigma_z)

print(sigma2_x,"\n\n", sigma2_y,"\n\n",sigma2_z,"\n\n")

a = sigma_x.trace()
b = sigma_y.trace()
c = sigma_z.trace()

print("Trace of σx,σy and σz is", a,",",b,",",c,"\n\n")


'''Ans 4'''


#3rd rank tensor 
def epsilon_(i,j=0,k=0):
	if i==j or j==k or i==k:
		return(0)
	elif [i,j,k] in [[1,2,3],[2,3,1],[3,1,2]]:
		return(1)
	elif [i,j,k] in [[1,3,2],[2,1,3],[3,2,1]]:
		return(-1)
	    
#kroneckar_delta	
def delta_(i,j):
	if i==j:
		return(1)
	else:
		return(0)

def find_index(x,y):
	for m in [1,2,3]:
		if np.array_equal(x,sigma_refer[m]):
			i=m
		if np.array_equal(y,sigma_refer[m]):
			j=m
	k=[num for num in [1,2,3] if num!=i and num!=j][0]
	return(i,j,k)

#commutator
def com(input_x,input_y):
	i,j,k=find_index(input_x,input_y)
	return(2*complex()*epsilon_(i,j,k)*sigma_refer[k])

def anti_com(input_x,input_y):
	i,j,k=find_index(input_x,input_y)
	return(2*delta_(i,j)*sigma_x.dot(sigma_x))

print('commutator of σx and σy',com(sigma_x,sigma_y),"\n")
print('anti-commutator of σx and σy',anti_com(sigma_y,sigma_y),"\n")

    
'''Ans 5'''
    

ladder_1 =(1/2*(sigma_x+(complex(0,1)*sigma_y)))
ladder_2 =(1/2*(sigma_x-(complex(0,1)*sigma_y)))

print(np.array_equal(ladder_2.dot(psi_1).reshape(2,1),psi_2))
print(np.array_equal(ladder_1.dot(psi_2).reshape(2,1),psi_1))
print("\n")	



