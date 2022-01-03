#https://leetcode.com/problems/unique-paths-ii/

def find(grid, m, n):
    if m == 0 or n == 0:
        return 1
    ret = 0
    if grid[m-1][n] == 0:
        ret += find(grid, m-1, n)
    if grid[m][n-1] == 0:
        ret += find(grid, m, n-1)
    return ret

def uniquePathsWithObstacles(obstacleGrid):
    return find(obstacleGrid, len(obstacleGrid)-1, len(obstacleGrid[0])-1)

i = [[1, 0]]
print(uniquePathsWithObstacles(i))