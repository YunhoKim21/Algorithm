def find(n):
    i=0
    while True:
        if n - 3*i <0:
            return -1
        if (n-3*i)%5==0:
            return int((n-3*i)/5+i)
        i+= 1
        
n = int(input())
print(find(n))