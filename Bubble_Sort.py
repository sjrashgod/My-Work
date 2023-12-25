#Method 1
def BubbleSort(arr):
    n = len(arr)
    
    for i in range(n-1):
        
        for j in range(0,n-i-1):
            
            if arr[j] > arr[j+1]:
                arr[j],arr[j+1] = arr[j+1],arr[j]
                
arr = [23,56,16,83,45,32,55]
print("Original Array")
print(arr)
BubbleSort(arr)

print("Modified sorted array is:")
print(arr)

#Method 2
def BubbleSort(arr):
    n = len(arr)
    
    for i in range(n-1):
        
        for j in range(0,n-i-1):
            
            if arr[j] > arr[j+1]:
                arr[j],arr[j+1] = arr[j+1],arr[j]

length = int(input('Total elements you will enter in the array:'))
arr=[]
for i in range(1,length+1):
    print('Entry Number',i,':')
    a=int(input('enter entry:'))
    arr.append(a)
    
print("Original Array")
print(arr)
BubbleSort(arr)

print("Modified sorted array is:")
print(arr)   

                