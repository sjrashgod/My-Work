#Method 1
array = [15,25,13,48,11]
print(array)
for i in range(len(array)):
         minimum_value = min(array[i:])
         minimum_index = array.index(minimum_value)
         array[i], array[minimum_index] = array[minimum_index],array[i]
print(array)

#Method 2 
import numpy as np
def sort(nums,n):
    for i in range(n-1):
        minpos=i
        for j in range(i,n):
            if nums[j]<nums[minpos]:
                minpos=j
        temp=nums[i]
        nums[i]=nums[minpos]
        nums[minpos]=temp
    return nums
nums=[]
n=int(input('Enter Length of Array You Want:'))
for i in range(1,n+1):
    print('Entry Number',i,':')
    a=int(input('Enter Number:'))
    nums.append(a)
    
print('Original Array')
print(np.array(nums)) 

print('New Array')
sort(nums,n)
print(np.array(nums))
