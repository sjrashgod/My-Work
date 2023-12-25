def BubbleSort(arr):
    n = len(arr)
    
    for i in range (n-1):
        
        for j in range (0, n-i-1):
            
            if arr[j] > arr[j+1] :
                arr[j], arr[j+1] = arr[j+1],arr[j]
                
arr = [23,56,16,83,45,32,55]
print("Original Array")
print(arr)
BubbleSort(arr)

print("Modified sort array is:")
print(arr)                
