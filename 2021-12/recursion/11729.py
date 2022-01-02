def hanoi(f, b, t, n):
    if n==1:
        print("{} {}".format(f, t))
        return
    hanoi(f, t, b, n-1)
    print("{} {}".format(f, t))
    hanoi(b, t, f, n-1)

hanoi(1, 2, 3, int(input()))
