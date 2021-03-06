from typing import List
from operator import length_hint
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x = [[3.5, 3], [3.1, 2.5], [-3, 3], [4, -2]]
x1 = [x[0][0], x[1][0], x[2][0], x[3][0]]
x2 = [x[0][1], x[1][1], x[2][1], x[3][1]]
alpha = 1
beta = 1
gamma = 0.5
delta = 0.5
m = [0, 0]


def f(v):
    x1 = v[0]
    x2 = v[1]
    z = math.pow(math.pow(x1, 2) + x2 - 11, 2) + math.pow(x1 + math.pow(x2,2) - 7, 2)    # Himmelblau-Funktion
    # z = 2 * math.pow(x1, 2) - 2 * (math.pow(x2, 2) - 11)
    return z


def sort(x):
    #xHelp = [x[0], x[1], x[2], x[3]]
    xHelp = x
    for i in [0, 1, 2, 3]:
        y = [f(x[0]), f(x[1]), f(x[2]), f(x[3])]
        y.sort()
        for j in [0, 1, 2, 3]:
            if f(x[i]) == y[j]:
                x[i], x[j] = x[j], x[i]     # swap

    return x


"""        xHelp[j] = x[j]     # xH3 = x3
           x[j] = x[i]
           x[i] = xHelp[j] """


def centre(x):
    # m = 1 / length_hint(x) * sum(x)  # Mittelpunkt
    for i in [0, 1, 2, 3]:
        m[0] += x[i][0]
        m[1] += x[i][1]
    m[0] = 1 / length_hint(x) * m[0]
    m[1] = 1 / length_hint(x) * m[1]
    return m


def iteration(m, x, e=list):
    r = [0, 0]
    r[0] = m[0] + alpha * (m[0] - x[3][0])  # reflexion
    r[1] = m[1] + alpha * (m[1] - x[3][1])
    if f(r) < f(x[0]):
        e[0] = m[0] + beta * (m[0] - x[3][0])   # expansion
        e[1] = m[1] + beta * (m[1] - x[3][1])
        if f(e) < f(r):
            x[3] = e
            print("Expansion: ", e)
        else:
            x[3] = r
            print("Reflexion: ", r)
    else:
        if f(r) < f(x[3]):
            x[3] = r
        if f(x[3]) > f(x[2]):
            r[0] = x[3][0] + gamma * (m[0] - x[3][0])   # contraction
            r[1] = x[3][1] + gamma * (m[1] - x[3][1])
            if f(r) < f(x[3]):
                x[3] = r
                print("Kontraktion: ", r)
            else:
                for i in [0, 1, 2, 3]:
                    if i != 0:
                        x[i][0] = delta * (x[i][0] + x[0][0])    # shrink
                        x[i][1] = delta * (x[i][1] + x[0][1])
                        print("Shrink: ", x[i])
    return x[0]


def main(x):
    #while (f(x[3]) - f(x[0]))/(abs(f(x[3])) + abs(f(x[0])) + 1) < math.pow(10, -15):     # math.pow(10, -15)
    z = [f(x[0]), f(x[1]), f(x[2]), f(x[3])]
    ax = plt.axes(projection='3d')
    ax.scatter(x1, x2, z, c='green')
#    plt.show()
    i = 0
    while i < 20:
        xtest = sort(x)
        mtest = centre(x)
        u = iteration(mtest, xtest)
        print("x: ", x)
        print(u)
        if i < 10:
            a = [x[0][0], x[1][0], x[2][0], x[3][0]]
            b = [x[0][1], x[1][1], x[2][1], x[3][1]]
            c = [f(x[0]), f(x[1]), f(x[2]), f(x[3])]
            ax.scatter(a, b, c, c='orange')
        i += 1
    ax.scatter(x[0][0], x[0][1], f(x[0]), c='red')
    plt.show()
    print(u)
    """
    print(x)
    xtest = sort(x)
    print(xtest)
    mtest = centre(x)
    print(mtest)
    u = iteration(mtest, xtest)
    print(u)
    """
    #print(u)
    #t = [[4, 2], [1, 1], [0, 3], [3, 3]]
    #s = centre(t)
    #s = [2, 2.25]
    #u = iteration(s, t)
    #u = s[1] + t[1][1]
    #print(u)


if __name__ == '__main__':
    main(x)


