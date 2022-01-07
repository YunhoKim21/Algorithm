n = int(input())
arr = []
for i in range(n):
    arr.append(int(input()))

def find(arr, n):
    if n == 1:
        return arr[0]
    if n == 2:
        return arr[0] + arr[1]
    m = [0 for i in range(n)]

    m[0] = arr[0]
    m[1] = arr[0] + arr[1]
    m[2] = max(arr[0], arr[1]) + arr[2]

    for i in range(3, n):
        m[i] = max(arr[i - 1] + m[i - 3], m[i - 2]) + arr[i]
    return m[n - 1]

print(find(arr, n))