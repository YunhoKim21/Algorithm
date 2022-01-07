n, target = map(int, input().split())
arr = []
for i in range(n):
    arr.append(int(input()))

arr = arr[::-1]
ret = 0
for i in range(n):
    num = int(target / arr[i])
    ret += num
    target -= num * arr[i]

print(ret)