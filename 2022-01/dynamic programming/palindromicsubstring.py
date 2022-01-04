def countSubstrings(s: str) -> int:
    even = []
    odd = []
    num = len(s)

    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            even.append(i)
    for i in range(len(s) - 2):
        if s[i] == s[i + 2]:
            odd.append(i)

    for i in even:
        k = 0
        while i-k>=0 and i+k+1<len(s) and s[i-k] == s[i+k+1]:
            k += 1
        k -= 1
        num += k+1
    for i in odd:
        k = 0

        while i - k >= 0 and i + k + 2 < len(s) and s[i - k] == s[i + k + 2]:
            k += 1
        k -= 1
        num += k+1
    return num

print(countSubstrings('abc'))