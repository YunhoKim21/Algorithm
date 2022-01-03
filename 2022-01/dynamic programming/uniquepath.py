#https://leetcode.com/problems/unique-paths/submissions/

def factorial(x):
    if x == 0:
        return 1
    return factorial(x-1) * x
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return int(factorial(m+n-2)/(factorial(n-1) * factorial(m-1)))

s = Solution()
print(factorial(2))