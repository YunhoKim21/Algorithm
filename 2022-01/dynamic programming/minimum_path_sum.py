def minPathSum(grid):
    m = len(grid)
    n = len(grid[0])
    cost = [[10000 for i in range(n)] for j in range(m)]
    cost[0][0] = grid[0][0]
    tovisit = [[0, 0]]
    for i in range(1, n):
        cost[0][i] = cost[0][i-1] + grid[0][i]
    for i in range(1, m):
        cost[i][0] = cost[i-1][0] + grid[i][0]
    if m == 1 or n == 1:
        return cost[m-1][n-1]
    for i in range(2, m+n-1):
        for k in range(1, i):
            point = [k, i-k]
            if k>=m or i-k>=n:
                continue
            cost[k][i-k] = min(cost[k-1][i-k], cost[k][i-k-1]) + grid[k][i-k]
    return cost[m-1][n-1]


print(minPathSum([[9, 1, 4, 8]]))
