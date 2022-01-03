#https://leetcode.com/problems/container-with-most-water/

def abs(x):
    if x>0:
        return x
    return -x

def area(height, i1, i2):
    return abs(i1 - i2) * min(height[i1], height[i2])

def maxArea(height) -> int:
    start = 0
    end = len(height) - 1
    ans = 0
    maxarea = 0
    while start<end:
        ans = area(height, start, end)
        if ans > maxarea:
            maxarea = ans
        if height[start] < height[end]:
            start += 1
        end -= 1
    return maxarea


print(maxArea([2, 3, 4, 5, 18, 17, 6]))