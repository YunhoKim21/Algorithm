def find(n, m, current):
    if len(current) == m:
        temp = [str(i+1) for i in current]
        print(' '.join(temp))
        return
    for i in range(n):
        flag = True
        for j in current:
            if j>i:
                flag = False
        if flag:
            find(n, m, current +[i])

n, m = input().split()
n = int(n)
m = int(m)

find(n, m, [])