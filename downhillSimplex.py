from typing import List
from operator import length_hint
import math as m

x = [[4, 2], [1, 1], [0, 3], [3, 3]]
alpha = 1
beta = 1
gamma = 0.5
delta = 0.5
m = [0, 0]


def f(v):
    z = m.pow(m.pow(v[1], 2) + v[2] - 11, 2) + m.pow(v[1] + m.pow(v[2],2) - 7, 2)    # Himmelblau-Funktion
    return z


def sort():
    xHelp = [x[0], x[1], x[2], x[3]]
    for i in [0, 1, 2, 3]:
        y = [f(x[0]), f(x[1]), f(x[2]), f(x[3])]
        y.sort()
        for j in [0, 1, 2, 3]:
            if f(x[i]) == y[j]:
                xHelp[j] = x[j]
                x[j] = x[i]
                x[i] = xHelp[j]


def centre():
    # m = 1 / length_hint(x) * sum(x)  # Mittelpunkt
    m = [0, 0]
    for i in [0, 1, 2, 3]:
        m[0] += x[i][0]
        m[1] += x[i][1]
    m[0] = 1 / length_hint(x) * m[0]
    m[1] = 1 / length_hint(x) * m[1]
    return m


def iteration():
    r = m + alpha * (m - x[3])  # reflexion
    if f(r) < f(x[0]):
        e = m + beta * (m - x[3])   # expansion
        if f(e) < f(r):
            x[3] = e
        else:
            x[3] = r
    else:
        if f(r) < f(x[3]):
            x[3] = r
        if f(x[3]) > f(x[2]):
            r = x[3] + gamma * (m - x[3])
            if f(r) < f(x[3]):
                x[3] = r;
            else:
                for i in [0, 1, 2, 3]:
                    if i != 0:
                        x[i] = delta * (x[i] + x[0])




