n = int(input())
arr = [[0 for i in range(10)] for j in range(n)]
arr[0] = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
M = 1000000000
for i in range(1, n):
    temp = [0 for j in range(10)]
    temp[0] = arr[i - 1][1]
    temp[9] = arr[i - 1][8]
    for j in range(1, 9):
        temp[j] = (arr[i - 1][j - 1]%M + arr[i - 1][j + 1]%M)%M
    arr[i] = temp

print(sum(arr[n - 1])%M)