#https://leetcode.com/problems/longest-palindromic-substring/submissions/
def longestPalindrome(s: str) -> str:
    even = []
    odd = []
    maxlen = 1
    maxseq = s[0]

    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            even.append(i)
    for i in range(len(s) - 2):
        if s[i] == s[i + 2]:
            odd.append(i)
    if len(even) == 0 and len(odd) == 0:
        return s[0]

    for i in even:
        k = 0
        while i-k>=0 and i+k+1<len(s) and s[i-k] == s[i+k+1]:
            k += 1
        k -= 1
        if (k + 1) * 2 > maxlen:
            maxlen = (k + 1) * 2
            maxseq = s[i-k:i+k+2]
    for i in odd:
        k = 0

        while i - k >= 0 and i + k + 2 < len(s) and s[i - k] == s[i + k + 2]:
            k += 1
        k -= 1
        if k*2 + 3 > maxlen:
            maxlen = k * 2 + 3
            maxseq = s[i - k: i+k+3]

    return maxseq

print(longestPalindrome('abba'))