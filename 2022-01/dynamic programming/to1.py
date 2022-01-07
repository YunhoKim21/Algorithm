n = int(input())

m = [0 for i in range(n + 1)]

def find(n):
    if m[n] != 0:
        return m[n]
    if n == 1:
        return 0
    can = []
    if n % 2 == 0:
        can.append(find(int(n / 2)) + 1)
    if n % 3 == 0:
        can.append(find(int(n / 3)) + 1)
    can.append(find(n - 1) + 1)
    m[n] = min(can)
    return min(can)

print(find(n))

