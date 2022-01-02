data = [[-1, -1] for i in range(41)]
data[0] = [1, 0]
data[1] = [0, 1]

def fibo(n):
    global data
    if data[n][0] != -1:
        return data[n]
    a = fibo(n-1)
    b = fibo(n-2)
    temp = [a[0] + b[0], a[1]+ b[1]]
    data[n] = temp
    return temp

n = int(input())
q = [-1 for i in range(n)]
for i in range(n):
    q[i] = int(input())

for i in range(n):
    print("{} {}".format(fibo(q[i])[0], fibo(q[i])[1]))