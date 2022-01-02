import sys

sys.setrecursionlimit(1000000000)

mem = [-1 for i in range(1000000)]
mem[1] = 1
mem[0] = 1  
def find(n):
    if mem[n]!= -1:
        return mem[n]
    res = (find(n-1) + find(n-2))%15746
    mem[n] = res
    return res

n = int(input())
print(find(n))