import copy

n = int(input())
arr = []
for i in range(n):
    arr.append(list(map(int, input().split())))
arr.sort(key = lambda x : x[1])
ret = 0
while len(arr) > 0:
    meeting = arr.pop(0)
    ret += 1
    copyarr = copy.deepcopy(arr)
    arr = [el for el in copyarr if el[0] >= meeting[1]]
print(ret)