n = int(input())
arr = []
for i in range(n):
    arr.append(list(map(int, input().split())) + [-1] * (n - i - 1))

m = [[0 for i in range(n)] for j in range(n)]
m[0] = arr[0]
for i in range(1, n):
    m[i][0] = m[i - 1][0] + arr[i][0]
    for j in range(1, n):
        m[i][j] = max(m[i-1][j],m[i - 1][j - 1]) + arr[i][j]
    
print(max(m[n-1]))

