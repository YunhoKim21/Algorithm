n = int(input())
arr = []
for i in range(n):
    arr.append(int(input()))

def find(arr, n):
    if n == 1:
        return arr[0]
    if n == 2:
        return arr[0] + arr[1]
    if n == 3:
        return sum(arr[0:3]) - min(arr[0:3])
    m = [0 for i in range(n)]
    m[0] = arr[0]
    m[1] = arr[0] + arr[1]
    m[2] = sum(arr[0:3]) - min(arr[0:3])

    for i in range(3, n):
        can = []
        can.append(m[i - 3] + arr[i - 1] + arr[i])
        can.append(m[i - 2] + arr[i])
        can.append(m[i - 1])
        m[i] = max(can)
    return m[n - 1]

print(find(arr, n))