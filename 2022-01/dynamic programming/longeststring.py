def lengthOfLongestSubstring(s: str) -> int:
    maxseq = ''
    maxlen = 0

    for i in range(len(s)):
        if s[i] in maxseq:
            maxseq = maxseq[maxseq.find(s[i]) + 1: ] + s[i]
        else:
            maxseq += s[i]
        if len(maxseq) > maxlen:
            maxlen = len(maxseq)
    return maxlen

print(lengthOfLongestSubstring('abcabcaa'))
