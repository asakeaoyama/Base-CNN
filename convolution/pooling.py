def pooling(pdata):
    # pdata formate: [y][x][i = conv times(16)]
    poolingOut = [[[0 for i in range(times)] for x in range(width)] for y in range(height)]