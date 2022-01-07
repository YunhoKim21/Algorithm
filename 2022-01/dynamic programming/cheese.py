import copy

a, b = map(int, input().split())
arr = []
for i in range(a):
    arr.append(list(map(int, input().split())))

def available(i, j):
    if i>=a or i<0 or j>=b or j < 0:
        return False
    return True
    
def simulate(arr, t):
    search = [[0, 0]]
    arr[0][0] = - t
    while len(search) > 0:
        node = search.pop(0)
        adj = [[node[0] + 1, node [1]], [node[0] - 1, node[1]], [node[0], node[1] + 1], [node[0], node[1] - 1]]
        for j in adj:
            if available(j[0], j[1]) and arr[j[0]][j[1]] != -t and arr[j[0]][j[1]] != 1:
                arr[j[0]][j[1]] = - t
                search.append(j)    
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            node = [i, j]
            adj = [[node[0] + 1, node [1]], [node[0] - 1, node[1]], [node[0], node[1] + 1], [node[0], node[1] - 1]]
            flag = False
            copyarray = copy.deepcopy(arr)

            for k in adj:
                if available(k[0], k[1]) and arr[k[0]][k[1]] == -t:
                    flag = True
            if flag:
                copyarray[i][j] = -100
    printarray(copyarray)
    arr = copyarray
                    
def printarray(arr):
    for i in arr:
        for j in i:
            print('{:3d}'.format(j), end = ' ')
        print()
printarray(arr)
simulate(arr, 1)
printarray(arr)