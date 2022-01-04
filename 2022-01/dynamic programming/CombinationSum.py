import time
def find(candidate, target):
    #time.sleep(1)
    print(candidate, target)
    if target == 0:
        return []
    if len(candidate) == 1:
        if target%(candidate[0]) == 0:
            return [candidate[0]] * int(target / candidate[0])
        else:
            return False
    
    ret = []
    i = 0
    t = candidate.pop()
    while target - i*t>=0:

        recursion = find(candidate, target - i*t)
        if type(recursion) != type(True):
            print(recursion + [t] * i)
            ret.append(recursion + [t] * i)
        
        i += 1
    
    return ret
def combinationSum(candidates, target):
    return find(candidates, target)

print(combinationSum([2, 3, 6, 7], 7))