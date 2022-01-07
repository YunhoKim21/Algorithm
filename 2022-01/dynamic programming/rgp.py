n = int(input())
arr = []

for i in range(n):
    arr.append(list(map(int, input().split())))
m = [[0, 0, 0] for i in range(n)]
m[0] = arr[0]
for i in range(1, n):
    m[i][0] = min(m[i - 1][1], m[i - 1][2]) + arr[i][0]
    m[i][1] = min(m[i - 1][0], m[i - 1][2]) + arr[i][1]
    m[i][2] = min(m[i - 1][1], m[i - 1][0]) + arr[i][2]

print(min(m[n - 1]))