import csv
from tkinter import *
from tkinter import ttk, messagebox

from operator import length_hint
import math
import os

"""
GUI Implementation
"""

# variable
x0 = [0, 0]  # initial guess
check_float = None  # check if a float-value is added to the input field


# optimize_clicked function (is called when the user clicks on the "Optimize!" Button)
def optimize_clicked():
    global check_float
    global selection
    selection = function_Menu.get()  # selection of a optimization function
    try:
        x0[0] = float(entry_x1.get())  # x1 start value
        x0[1] = float(entry_x2.get())  # x2 start value
        check_float = True
    except:
        messagebox.showerror("Type Error", "The x1/x2 - values must be of the type float. Please try again.")
        check_float = None
    if check_float:
        root.destroy()  # automatically closes the window if the algorithm finishes successfully


# Root & UI
root = Tk()
root.title('Nelder Mead Visualizer')
root.resizable(0, 0)

# Canvas
canvas_gui = Canvas(root, width=750, height=300)
canvas_gui.pack()

# Label Title
label_title = Label(root, text='Visualization of the Nelder Mead Algorithm')
label_title.config(font=('Arial', 20))
canvas_gui.create_window(375, 40, window=label_title)

# Label Explanation
label_explanation_1 = Label(root,
                            text='This program applies the Nelder Mead Algorithm to a chosen objective function and '
                                 'visualizes the results.')
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

# Optimize Button
button_optimize = Button(root, font=('Arial', 13), command=optimize_clicked, text='Optimize!', bg='lightcoral')
canvas_gui.create_window(375, 260, window=button_optimize)

root.mainloop()

"""
Algorithm Implementation
"""
# Initialization of variables
x = [[0, 0], [0, 0], [0, 0], [0, 0]]
alpha = 1
beta = 2
gamma = 0.5
delta = 0.5
m = [0, 0]

"""
Initial Simplex for Nelder Mead
"""


# Spendley´s et al regular simplex
def spendley_regular_simplex(x0_in, size):
    p = 1 / (3 * math.sqrt(2)) * (3 - 1 + math.sqrt(3 + 1))
    q = 1 / (3 * math.sqrt(2)) * (math.sqrt(3 + 1) - 1)
    x[0] = x0_in

    for i in [1, 2, 3]:
        for j in [0, 1]:
            if i - 1 == j:
                x[i][j] = x[0][j] + size * p
            else:
                x[i][j] = x[0][j] + size * q
    return x


# Initialization with Random Bounds
def random_bounds(lb, ub):
    theta = [0.2, 0.4, 0.6, 0.8]
    for i in [0, 1, 2, 3]:
        for j in [0, 1]:
            x[i][j] = lb[j] + theta[i] * (ub[j] - lb[j])
    return x


"""
Steps for Algorithm Iteration
"""


# Optimization Function
def f(v):
    if selection == "Himmelblau":
        z = math.pow(math.pow(v[0], 2) + v[1] - 11, 2) + math.pow(v[0] + math.pow(v[1], 2) - 7,
                                                                  2)  # Himmelblau-Funktion
    else:
        z = math.pow(1 - v[0], 2) + 100 * math.pow((v[1] - math.pow(v[0], 2)), 2)  # Rosenbrock-Funktion

    return z


# Sort x-values according to their function value, ascending
def sort(x):
    for i in [0, 1, 2, 3]:
        y = [f(x[0]), f(x[1]), f(x[2]), f(x[3])]  # calculate function values
        y.sort()  # sort f-values ascending
        for j in [0, 1, 2, 3]:
            if f(x[i]) == y[j]:
                x[i], x[j] = x[j], x[i]  # swap values according to function values
    return x


# Calculate centre of x1 and x2 values
def centre(data):
    for i in [0, 1, 2]:
        m[0] += data[i][0]
        m[1] += data[i][1]
    m[0] = 1 / (length_hint(data) - 1) * m[0]
    m[1] = 1 / (length_hint(data) - 1) * m[1]
    return m


# Iteration of the algorithm
def iteration(m, x):
    r = [0, 0]
    h = [0, 0]
    c = [0, 0]
    e = [0, 0]
    r[0] = m[0] + alpha * (m[0] - x[3][0])  # reflexion
    r[1] = m[1] + alpha * (m[1] - x[3][1])
    if f(r) < f(x[0]):
        e[0] = m[0] + beta * (m[0] - x[3][0])  # expansion
        e[1] = m[1] + beta * (m[1] - x[3][1])
        if f(e) < f(r):
            x[3] = e
        else:
            x[3] = r
    else:
        if f(r) < f(x[2]):
            x[3] = r
        else:
            if f(x[3]) < f(r):
                h[0] = x[3][0]
                h[1] = x[3][1]
            else:
                h[0] = r[0]
                h[1] = r[1]
            c[0] = h[0] + gamma * (m[0] - h[0])  # contraction
            c[1] = h[1] + gamma * (m[1] - h[1])
            if f(c) < f(x[3]):
                x[3] = c
            else:
                for i in [0, 1, 2, 3]:
                    if i != 0:
                        x[i][0] = x[i][0] + delta * (x[0][0] - x[i][0])  # Shrink
                        x[i][1] = x[i][1] + delta * (x[0][1] - x[i][1])
    return x[0]


def main():
    global selection
    if check_float == None:
        return None

    x = spendley_regular_simplex(x0, 0.7)  # initialization of the simplex

    header = ["x1", "x2", "f(x1, x2)", "Iteration", "Algorithmus"]  # header of the csv-sheet

    row1 = [x[0][0], x[0][1], f(x[0]), '0', selection]
    row2 = [x[1][0], x[1][1], f(x[1]), '0', selection]
    row3 = [x[2][0], x[2][1], f(x[2]), '0', selection]
    row4 = [x[3][0], x[3][1], f(x[3]), '0', selection]

    # write x, y, z values to csv sheets:
    with open('nelder_mead.csv', 'w') as cs:
        write = csv.writer(cs)
        write.writerow(header)
        write.writerow(row1)
        write.writerow(row2)
        write.writerow(row3)
        write.writerow(row4)

    i = 1
    # alternativ: while (f(x[3]) - f(x[0]))/(abs(f(x[3])) + abs(f(x[0])) + 1) < math.pow(10, -15):
    while i < 20:
        xtest = sort(x)  # sort x-values according to function values
        mtest = centre(x)  # calculate centre
        u = iteration(mtest, xtest)  # iteration of the algorithm
        if i < 20:
            row1 = [x[0][0], x[0][1], f(x[0]), str(i), selection]
            row2 = [x[1][0], x[1][1], f(x[1]), str(i), selection]
            row3 = [x[2][0], x[2][1], f(x[2]), str(i), selection]
            row4 = [x[3][0], x[3][1], f(x[3]), str(i), selection]

            with open('nelder_mead.csv', 'a') as cs:
                write = csv.writer(cs)
                write.writerow(row1)
                write.writerow(row2)
                write.writerow(row3)
                write.writerow(row4)

        i += 1


# call the main function
if __name__ == '__main__':
    main()
    exec(open('CSV_Visualization_v2.py').read())