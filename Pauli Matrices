import numpy as n
from subprocess import call 
import cmath
import os
psi_1=n.array([[1],[0]],n.int32)
psi_2=n.array([[0],[1]],n.int32)

sigma_x=n.array([[0,1],[1,0]])
sigma_y=n.array([[0,complex(0,-1)],[complex(0,1),0]])
sigma_z=n.array([[1,0],[0,-1]])
sigma_refer={1:sigma_x,2:sigma_y,3:sigma_z}

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
		if n.array_equal(x,sigma_refer[m]):
			i=m
		if n.array_equal(y,sigma_refer[m]):
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

#Compute ladder operators
ladder_neg=(1/2*(sigma_x-(complex(0,1)*sigma_y)))
ladder_pos=(1/2*(sigma_x+(complex(0,1)*sigma_y)))

#Compute σ2x,σ2y,σ2z and Trace of σx,σy and σz
'''def os.system('cls'):
    # check and make call for specific operating system 
    _ = call('clear' if os.name =='posix' else 'cls') '''
while 1:
	print("Welcome to the program that outputs the answers to the questions of Assignment_1:\n Type the question number(1,2,3 or 4) in input for the corresponding answer :) \n\n\n ")
	user_inpt1=int(input("Enter(1,2,3,4,5) : "))
	if user_inpt1 in [1,2,3,4,5]:
		if user_inpt1==1:
			print("seriously dude? You want me to print possible spins of electron?")
			print(psi_1,'\n',psi_2)
			input('')
			os.system('cls')
			continue
		elif user_inpt1==2:
			print("seriously dude? Now YOu want me to print pauli matrices?")
			print(sigma_x,sigma_y,sigma_z)
			input('')
			os.system('cls')
			continue
		elif user_inpt1==3:
			print("\n\nsigma^2_x : \n",sigma_x.dot(sigma_x),"\nsigma^2_y : \n",sigma_y.dot(sigma_y),"\nsigma^2_z \n",sigma_z.dot(sigma_z))
			print("\n\nTrace of σx,σy and σz is ",sigma_x.trace()," ",sigma_y.trace()," and ",sigma_z.trace())
			input('')
			continue
		elif user_inpt1==4:
			print('commutor of σx and σy',com(sigma_x,sigma_y))
			print('anti-commutor of σx and σy',anti_com(sigma_x,sigma_y))
			input('')
			os.system('cls')
			continue
		elif user_inpt1==5:
			#check
			print('\n\nIt is ',n.array_equal(ladder_neg.dot(psi_1).reshape(2,1),psi_2),' that (σ−)(ψ1)=(ψ2)')
			print('\n\nIt is ',n.array_equal(ladder_pos.dot(psi_2).reshape(2,1),psi_1),' that (σ+)(ψ2)=(ψ1)\n\n')
                        input('')
			os.system('cls')
			continue
		else:
			break
