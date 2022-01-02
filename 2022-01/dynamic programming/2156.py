def find(array):
    if len(array)<3:
        return sum(array)
    return max(find(array[1:]), array[0] + find(array[2:]), array[0]+array[1]+find(array[3:]))

n = int(input())
arr = [0 for i in range(n)]
for i in range(n):
    arr[i] = int(input())
print(find(arr))