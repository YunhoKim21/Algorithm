count = 0

def isattack(a, b, c, d):
    if a+b == c+d:
        return True
    if a-b == c-d:
        return True


def find(current, n):
    global count
    if len(current) == n:
        count += 1
        return

    for i in range(n):
        if i not in current:
            flag = True
            for j in range(len(current)):
                if isattack(j, current[j], len(current), i):
                    flag = False
            if flag:
                find(current + [i], n)

find([], int(input()))
print(count)