def cansum1(n, array): 
    if n < 0:
        return False
    if len(array) == 1:
        if array[0] == n:
            return True
        return False
    if array[0] == n:
        return True
    return cansum1(n-array[0], array[1:]) or cansum1(n, array[1:])

def cansum2(n, array):
    
    for i in array:
        if n == i:
            return True
        newlist = [j for j in array if j!=i]
        if cansum2(n-i, newlist):
            return True
    return False

print(cansum2(7, [5, 3, 10, 1]))