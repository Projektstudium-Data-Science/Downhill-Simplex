from tkinter import *
from tkinter import ttk
import time

from operator import length_hint
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

root = Tk()
root.title('Downhill Simplex Algorithm Visualisation')
root.maxsize(900, 600)
root.config(bg='black')

x_data = []
y_data = []
z_data = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line, = ax.plot(0, 0, 0)

x = [[0,0], [0,0], [0,0], [0,0], [0,0]]

z = [0, 0, 0, 0]
a1, b1, c1 = [x[0][0], x[1][0]], [x[0][1], x[1][1]], [z[0], z[1]]
a2, b2, c2 = [x[0][0], x[2][0]], [x[0][1], x[2][1]], [z[0], z[2]]
a3, b3, c3 = [x[0][0], x[3][0]], [x[0][1], x[3][1]], [z[0], z[3]]
a4, b4, c4 = [x[1][0], x[2][0]], [x[1][1], x[2][1]], [z[1], z[2]]
a5, b5, c5 = [x[1][0], x[3][0]], [x[1][1], x[3][1]], [z[1], z[3]]
a6, b6, c6 = [x[2][0], x[3][0]], [x[2][1], x[3][1]], [z[2], z[3]]
p1 = ax.plot(a1, b1, c1, color='green')
p2 = ax.plot(a2, b2, c2, color='green')
p3 = ax.plot(a3, b3, c3, color='green')
p4 = ax.plot(a4, b4, c4, color='green')
p5 = ax.plot(a5, b5, c5, color='green')
p6 = ax.plot(a6, b6, c6, color='green')



#fig, ax = plt.axes(projection='3d')

#variables
selected_func = StringVar()
x0 = [0, 0]


def drawData(data):
    c_height = 380
    c_width = 600


def generate():
    print('Function Selected: ', selected_func.get())
    try:
        x0[0] = int(x0Entry.get())
    except:
        x0[0] = 3.1
    try:
        x0[1] = int(x1Entry.get())
    except:
        x0[1] = 2.5
    print("x0 = ", x0)
    UI_frame.quit()


#frame
UI_frame = Frame(root, width=600, height=200, bg='grey')
UI_frame.grid(row=0, column=0, padx=10, pady=5)

canvas = Canvas(root, width=600, height=380)
# canvas.grid(row=1, column=0, padx=10, pady=5)

#User Interface Area
#Row[0]
Label(UI_frame, text="Function: ", bg='grey').grid(row=0, column=0, padx=5, pady=5, sticky=W)
funcMenu = ttk.Combobox(UI_frame, textvariable=selected_func, values=['Himmelblau', 'Standard Optimierungsproblem'])
funcMenu.grid(row=0, column=1, padx=5, pady=5)
funcMenu.current(0)
Button(UI_frame, text="Generate", command=generate).grid(row=0, column=2, padx=5, pady=5)
#Row[1]
Label(UI_frame, text="Startpunkt", bg='grey').grid(row=1, column=0, padx=5, pady=5, sticky=W)
Label(UI_frame, text="x1: ", bg='grey').grid(row=1, column=1, padx=5, pady=5, sticky=W)
x0Entry = Entry(UI_frame)
x0Entry.grid(row=1, column=2, padx=5, pady=5, sticky=W)
Label(UI_frame, text="x2: ", bg='grey').grid(row=1, column=3, padx=5, pady=5, sticky=W)
x1Entry = Entry(UI_frame)
x1Entry.grid(row=1, column=4, padx=5, pady=5, sticky=W)

root.mainloop()

x = [[0, 0], [0, 0], [0, 0], [0, 0]]
alpha = 1
beta = 2
gamma = 0.5
delta = 0.5
m = [0, 0]


def spendley_regular_simplex(x0_in, l):
    p = 1/(3*math.sqrt(2)) * (3 - 1 + math.sqrt(3 + 1))
    q = 1/(3*math.sqrt(2)) * (math.sqrt(3 + 1) - 1)
    x[0] = x0_in

    for i in [1, 2, 3]:
        for j in [0, 1]:
            if i - 1 == j:
                x[i][j] = x[0][j] + l * p
            else:
                x[i][j] = x[0][j] + l * q
    return x


def random_bounds(lb, ub):
    theta = [0.2, 0.4, 0.6, 0.8]
    for i in [0, 1, 2, 3]:
        for j in [0, 1]:
            x[i][j] = lb[j] + theta[i] * (ub[j] - lb[j])
    return x


def f(v):
    z = math.pow(math.pow(v[0], 2) + v[1] - 11, 2) + math.pow(v[0] + math.pow(v[1], 2) - 7, 2)    # Himmelblau-Funktion
    # z = 2 * math.pow(x1, 2) - 2 * (math.pow(x2, 2) - 11)
    #z = math.pow(v[0] - 6, 2) + 2 * math.pow(v[1] - 3, 2)
    return z


def sort(x):
    #xHelp = [x[0], x[1], x[2], x[3]]
    for i in [0, 1, 2, 3]:
        y = [f(x[0]), f(x[1]), f(x[2]), f(x[3])]
        y.sort()
        for j in [0, 1, 2, 3]:
            if f(x[i]) == y[j]:
                x[i], x[j] = x[j], x[i]     # swap

    return x


def centre(data):
    # m = 1 / length_hint(x) * sum(x)  # Mittelpunkt
    for i in [0, 1, 2]:
        m[0] += data[i][0]
        m[1] += data[i][1]
    m[0] = 1 / (length_hint(data)-1) * m[0]
    m[1] = 1 / (length_hint(data)-1) * m[1]
    return m


def iteration(m, x):
    r = [0, 0]
    h = [0, 0]
    c = [0, 0]
    e = [0, 0]
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
        if f(r) < f(x[2]):
            x[3] = r
            print("Reflexion: ", r)
        else:
            if f(x[3]) < f(r):
                h[0] = x[3][0]
                h[1] = x[3][1]
            else:
                h[0] = r[0]
                h[1] = r[1]
            c[0] = h[0] + gamma * (m[0] - h[0])
            c[1] = h[1] + gamma * (m[1] - h[1])
            if f(c) < f(x[3]):
                x[3] = c
                print("Kontraktion: ", c)
            else:
                for i in [0, 1, 2, 3]:
                    if i != 0:
                        x[i][0] = x[i][0] + delta * (x[0][0] - x[i][0])
                        x[i][1] = x[i][1] + delta * (x[0][1] - x[i][1])
                        print("Komprimierung: ", x)
    return x[0]


def update_plot(frame):
    ax.clear()
    global p1
    p1 = ax.plot(a1, b1, c1, color='green')
    """
    global p1, p2, p3, p4, p5, p6
    
    p1.remove()
    p2.remove()
    p3.remove()
    p4.remove()
    p5.remove()
    p6.remove()
    
    z = [f(x[0]), f(x[1]), f(x[2]), f(x[3])]
    a1, b1, c1 = [x[0][0], x[1][0]], [x[0][1], x[1][1]], [z[0], z[1]]
    a2, b2, c2 = [x[0][0], x[2][0]], [x[0][1], x[2][1]], [z[0], z[2]]
    a3, b3, c3 = [x[0][0], x[3][0]], [x[0][1], x[3][1]], [z[0], z[3]]
    a4, b4, c4 = [x[1][0], x[2][0]], [x[1][1], x[2][1]], [z[1], z[2]]
    a5, b5, c5 = [x[1][0], x[3][0]], [x[1][1], x[3][1]], [z[1], z[3]]
    a6, b6, c6 = [x[2][0], x[3][0]], [x[2][1], x[3][1]], [z[2], z[3]]
    p1 = ax.plot(a1, b1, c1, color='green')
    p2 = ax.plot(a2, b2, c2, color='green')
    p3 = ax.plot(a3, b3, c3, color='green')
    p4 = ax.plot(a4, b4, c4, color='green')
    p5 = ax.plot(a5, b5, c5, color='green')
    p6 = ax.plot(a6, b6, c6, color='green')
    fig.canvas.draw_idle()
    """


def main():
    #while (f(x[3]) - f(x[0]))/(abs(f(x[3])) + abs(f(x[0])) + 1) < math.pow(10, -15):     # math.pow(10, -15)
    x = spendley_regular_simplex(x0, 0.7)
    print(x)
    #x = random_bounds([-10, -8], [10, 8])
    #x = random_bounds([0, 1], [6, 5])
    #x = [[3.5, 3], [3.1, 2.5], [-3, 3], [4, -2]]
    x1 = [x[0][0], x[1][0], x[2][0], x[3][0]]
    x2 = [x[0][1], x[1][1], x[2][1], x[3][1]]
    z = [f(x[0]), f(x[1]), f(x[2]), f(x[3])]

    a1, b1, c1 = [x[0][0], x[1][0]], [x[0][1], x[1][1]], [z[0], z[1]]
    a2, b2, c2 = [x[0][0], x[2][0]], [x[0][1], x[2][1]], [z[0], z[2]]
    a3, b3, c3 = [x[0][0], x[3][0]], [x[0][1], x[3][1]], [z[0], z[3]]
    a4, b4, c4 = [x[1][0], x[2][0]], [x[1][1], x[2][1]], [z[1], z[2]]
    a5, b5, c5 = [x[1][0], x[3][0]], [x[1][1], x[3][1]], [z[1], z[3]]
    a6, b6, c6 = [x[2][0], x[3][0]], [x[2][1], x[3][1]], [z[2], z[3]]
    ax.plot(a1, b1, c1, color='green')
    ax.plot(a2, b2, c2, color='green')
    ax.plot(a3, b3, c3, color='green')
    ax.plot(a4, b4, c4, color='green')
    ax.plot(a5, b5, c5, color='green')
    ax.plot(a6, b6, c6, color='green')

    # ax.scatter(x1, x2, z, c='green')
    # plt.plot(x1, x2, z, c='green')
    i = 0
    print(x)
    while i < 20:
        xtest = sort(x)
        mtest = centre(x)
        u = iteration(mtest, xtest)
        print("x: ", x)
        print(u)
        if i < 10:
            #a = [x[0][0], x[1][0], x[2][0], x[3][0]]
            #b = [x[0][1], x[1][1], x[2][1], x[3][1]]
            #c = [f(x[0]), f(x[1]), f(x[2]), f(x[3])]
            # time.sleep(0.2)   # Sleep funktioniert nur für Berechnung aber nicht für Darstellung
            #ax.scatter(a, b, c, c='orange')
            #time.sleep(0.2)
         animation = FuncAnimation(fig, func=update_plot, frames=10, interval=10)

        i += 1
    ax.scatter(x[0][0], x[0][1], f(x[0]), c='red')
    plt.show()      # eventuell den plot in while-Schleife und dann jedes Mal quitten und neu starten
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
    main()