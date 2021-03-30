import csv
from tkinter import *
from tkinter import ttk, messagebox
import time

from operator import length_hint
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# GUI Implementation

# variable
x0 = [0, 0]  # initial guess


# optimize function
def optimize():
    global check
    check = None
    global selection
    selection = function_Menu.get()
    try:
        x0[0] = float(entry_x1.get())
        x0[1] = float(entry_x2.get())
        check = True
    except:
        messagebox.showerror("Type Error", "The x1/x2 - values must be of the type float. Please try again")
        check = None
    if(check):
        root.destroy()


# root & UI
root = Tk()
root.title('Nelder Mead Visualizer')
root.resizable(0, 0)

#canvas
canvas_gui = Canvas(root, width=750, height=300)
canvas_gui.pack()

# Label Title
label_title = Label(root, text='Visualization of the Nelder Mead Algorithm')
label_title.config(font=('Arial', 20))
canvas_gui.create_window(375, 40, window=label_title)

# Label Explanation
label_explanation_1 = Label(root,
                            text='This program applies the Nelder Mead Algorithm to a chosen objective function and visualizes the results.')
label_explanation_1.config(font=('Arial', 10))
canvas_gui.create_window(342, 80, window=label_explanation_1)

label_explanation_2 = Label(root, text='Please choose a function and enter a initial guess for x1/x2:')
label_explanation_2.config(font=('Arial', 10))
canvas_gui.create_window(213, 105, window=label_explanation_2)

# Label Function
label_function = Label(root, text='Function:')
label_function.config(font=('Arial', 10), bg='lavender')
canvas_gui.create_window(67, 170, window=label_function)

# Function Dropdown - Menu
function_Menu = ttk.Combobox(root, values=['Himmelblau', 'Rosenbrock'])
function_Menu.current(0)
canvas_gui.create_window(174, 170, window=function_Menu)

# Label x1 / x2
label_x1 = Label(root, text='x1 - value:')
label_x1.config(font=('Arial', 10), bg='lavender')
canvas_gui.create_window(420, 170, window=label_x1)

label_x2 = Label(root, text='x2 - value:')
label_x2.config(font=('Arial', 10), bg='lavender')
canvas_gui.create_window(420, 200, window=label_x2)

# Entry x1 / x2
entry_x1 = Entry(root, textvariable='x1_value')
entry_x1.config(bg='lavender')
canvas_gui.create_window(523, 170, window=entry_x1)

entry_x2 = Entry(root, textvariable='x2_value')
entry_x2.config(bg='lavender')
canvas_gui.create_window(523, 200, window=entry_x2)

# optimize button
button_optimize = Button(root, font=('Arial', 13), command=optimize, text='Optimize!', bg='lightcoral')
canvas_gui.create_window(375, 260, window=button_optimize)

root.mainloop()

# Algorithm Implementation

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
    if (selection == "Himmelblau"):
        z = math.pow(math.pow(v[0], 2) + v[1] - 11, 2) + math.pow(v[0] + math.pow(v[1], 2) - 7, 2)  #Himmelblau-Funktion
        print('HIMMELBLAU')
    elif (selection == 'Rosenbrock'):
        z = math.pow(1 - v[0], 2) + 100 * math.pow((v[1] - math.pow(v[0], 2)), 2)                    #Rosenbrock-Funktion
        print('ROSENBROCK')
    else:
        print("FEHLER")
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


def main():
    if(check == None):
        return None

    # while (f(x[3]) - f(x[0]))/(abs(f(x[3])) + abs(f(x[0])) + 1) < math.pow(10, -15):     # math.pow(10, -15)
    x = spendley_regular_simplex(x0, 0.7)
    print(x)
    # x = random_bounds([-10, -8], [10, 8])
    # x = random_bounds([0, 1], [6, 5])
    # x = [[3.5, 3], [3.1, 2.5], [-3, 3], [4, -2]]
    x1 = [x[0][0], x[1][0], x[2][0], x[3][0]]
    x2 = [x[0][1], x[1][1], x[2][1], x[3][1]]
    z = [f(x[0]), f(x[1]), f(x[2]), f(x[3])]

    header = ["x1", "x2", "f(x1, x2)", "Iteration"]

    row1 = [x[0][0], x[0][1], f(x[0]), '0']
    row2 = [x[1][0], x[1][1], f(x[1]), '0']
    row3 = [x[2][0], x[2][1], f(x[2]), '0']
    row4 = [x[3][0], x[3][1], f(x[3]), '0']

    with open('nelder_mead.csv', 'w') as cs:
        write = csv.writer(cs)
        write.writerow(header)
        write.writerow(row1)
        write.writerow(row2)
        write.writerow(row3)
        write.writerow(row4)

    ax = plt.axes(projection='3d')

    a1, b1, c1 = [x[0][0], x[1][0]], [x[0][1], x[1][1]], [z[0], z[1]]
    a2, b2, c2 = [x[0][0], x[2][0]], [x[0][1], x[2][1]], [z[0], z[2]]
    a3, b3, c3 = [x[0][0], x[3][0]], [x[0][1], x[3][1]], [z[0], z[3]]
    a4, b4, c4 = [x[1][0], x[2][0]], [x[1][1], x[2][1]], [z[1], z[2]]
    a5, b5, c5 = [x[1][0], x[3][0]], [x[1][1], x[3][1]], [z[1], z[3]]
    a6, b6, c6 = [x[2][0], x[3][0]], [x[2][1], x[3][1]], [z[2], z[3]]
    plt.plot(a1, b1, c1, color='green')
    plt.plot(a2, b2, c2, color='green')
    plt.plot(a3, b3, c3, color='green')
    plt.plot(a4, b4, c4, color='green')
    plt.plot(a5, b5, c5, color='green')
    plt.plot(a6, b6, c6, color='green')

    # ax.scatter(x1, x2, z, c='green')
    # plt.plot(x1, x2, z, c='green')
    i = 1
    print(x)
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
            # time.sleep(0.2)   # Sleep funktioniert nur für Berechnung aber nicht für Darstellung
            #plt.show()
            #ax.scatter(a, b, c, c='orange')
            ax.scatter(a, b, c, c=np.linalg.norm([a,b,c], axis=0))

            row1 = [x[0][0], x[0][1], f(x[0]), str(i)]
            row2 = [x[1][0], x[1][1], f(x[1]), str(i)]
            row3 = [x[2][0], x[2][1], f(x[2]), str(i)]
            row4 = [x[3][0], x[3][1], f(x[3]), str(i)]

            with open('nelder_mead.csv', 'a') as cs:
                write = csv.writer(cs)
                write.writerow(row1)
                write.writerow(row2)
                write.writerow(row3)
                write.writerow(row4)
            # time.sleep(0.2)

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