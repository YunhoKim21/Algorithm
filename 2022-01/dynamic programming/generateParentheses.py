def findstarting(x):
    ret = 0
    for i in x:
        if i == '(':
            ret += 1
    return ret

def findending(x):
    ret = 0
    for i in x:
        if i == ')':
            ret += 1
    return ret

def generateParenthesis(n: int):
    search = ['']
    ans = []
    while len(search) > 0:
        a = search.pop(0)
        if len(a) == 2*n:
            ans.append(a)
        if findstarting(a) < n:
            search.append(a + '(')
        if findending(a) < findstarting(a):
            search.append(a + ')')
    print(ans)

generateParenthesis(3)