n, k = map(int, input().split())
arr = []
for i in range(n):
    arr.append(list(map(int, input().split())))

data = [[0 for i in range(k)] for j in range(n)]
for i in range(k):
    if i + 1 >= arr[0][0]:
        data[0][i] = arr[0][1]

for i in range(1, n):
    mass = arr[i][0]; value = arr[i][1]
    for j in range(0, mass-1):
        data[i][j] = data[i - 1][j]
    for j in range(mass-1,  k):
        if j - mass >= 0:
            data[i][j] = max(data[i - 1][j], value + data[i - 1][j - mass])
        else:
            data[i][j] = max(data[i - 1][j], value)
    
print(data[n - 1][k - 1])