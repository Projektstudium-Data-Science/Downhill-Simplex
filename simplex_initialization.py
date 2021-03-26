import math


def axis_by_axis_simplex(x0,l):
    x = [[0,0], [0,0], [0,0],[0,0]]
    x[0] = x0

    for i in [1,2,3]:
        for j in [0,1]:
            if(i - 1 == j):
                x[i][j] = x[0][j] + l[j]
            else:
                x[i][j] = x[0][j]
    return x


def spendley_regular_simplex(x0,l):
    x = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p = 1/(3*math.sqrt(2)) * (3 - 1 + math.sqrt(3 + 1))
    q = 1/(3*math.sqrt(2)) * (math.sqrt(3 + 1) -1)
    x[0] = x0

    for i in [1,2,3]:
        for j in [0,1]:
            if(i - 1 == j):
                x[i][j] = x[0][j] + l* p
            else:
                x[i][j] = x[0][j] + l * q
    return x
