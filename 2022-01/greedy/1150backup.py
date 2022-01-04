#https://www.acmicpc.net/problem/1150

n, k = input().split()
n = int(n)
k = int(k)
cost = 0

p = []
for i in range(n):
    p.append(int(input()))

search = []
for i in range(n-1):
    search.append([p[i+1] - p[i], [i, i+1]])

for i in range(k):
    search.sort(key = lambda x : x[0])
    greedy = search.pop(0)
    cost += greedy[0]
    for j in greedy[1]:
        for k in range(len(search)):
            print(search[k][1])
            if j in search[k][1]:
                search.pop(k)
    fi = greedy[1][0]
    si = greedy[1][1]
    if si+1<n and fi-1>=0:
        search.append([p[si+1]-p[fi-1] - 2*(p[si]-p[fi]), [fi-1, si+1]])
print(cost)
