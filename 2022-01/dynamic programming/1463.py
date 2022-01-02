min = 0
def find(n, step):
    global min
    if n == 1:
        if step<min:
            min = step
        return
    if n%2 == 0:
        find(int(n/2), step+1)
    if n%3 == 0:
        find(int(n/3), step + 1)
    find(n-1, step + 1)

n = int(input())
min = n
find(n, 0)
print(min)