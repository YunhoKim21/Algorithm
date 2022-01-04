#https://leetcode.com/problems/longest-substring-without-repeating-characters/

def lengthOfLongestSubstring(s: str) -> int:
    current = ''
    maxlen = 0
    for i in range(len(s)):
        if s[i] in current:
            current = current[current.find(s[i])+1:]
            current += s[i]
        else:
            current += s[i]
            if len(current) > maxlen:
                maxlen = len(current)
    return maxlen
            
print(lengthOfLongestSubstring('pwwkew'))