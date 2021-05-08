import csv
import numpy as np
import math
import plotly.graph_objs as go
from tkinter import *
from tkinter import ttk, messagebox

"""
Initialization of variables ============================================================================================
"""

x0 = [[-5, -5], [5, 5]]  # initial bounds
x = [[0, 0], [0, 0], [0, 0], [0, 0]]

# GUI variables
check_float = None  # checks if a float-value is added to the input field
check_optimize_clicked = None  # checks if the optimize-Button is clicked

"""
Initial Simplex for Nelder Mead ========================================================================================
"""


# Initialization with Random Bounds
def random_bounds(lb, ub):
    theta = [0.2, 0.4, 0.6, 0.8]
    for i in [0, 1, 2, 3]:
        for j in [0, 1]:
            x[i][j] = lb[j] + theta[i] * (ub[j] - lb[j])
    return x


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


"""
GUI Implementation =====================================================================================================
"""


# is called when the function for the initial simplex is chosen
def function_chosen():
    global initial_selection
    initial_selection = initial_function.get()  # selection of the function for the initial simplex

    if initial_selection == 'Spendley´s Regular Simplex':

        # variables
        x0 = [0, 0]  # initial guess
        check_float = None  # check if a float-value is added to the input field

        # optimize_clicked function (is called when the user clicks on the "Optimize!" Button)
        def optimize_clicked():
            global check_optimize_clicked
            check_optimize_clicked = True
            global check_float
            global selection
            global x
            selection = function_Menu.get()  # selection of a optimization function
            try:
                x0[0] = float(entry_x.get())  # x start value
                x0[1] = float(entry_y.get())  # y start value
                check_float = True
            except:
                messagebox.showerror("Type Error", "The x/y - values must be of the type float. Please try again.")
                check_float = None
                check_optimize_clicked = None
            if check_float:
                x = spendley_regular_simplex(x0, 0.7)  # initialization of the simplex
                root.destroy()  # automatically closes the window if the algorithm finishes successfully
                init.destroy()

        # Root & UI for Spendley´s Regular Simplex
        root = Tk()
        root.title('Nelder Mead Visualizer')
        root.resizable(0, 0)

        # Canvas
        canvas_gui = Canvas(root, width=400, height=320)
        canvas_gui.pack()

        # Label explanation 4
        label_explanation_4 = Label(root, text='Your choice: Spendley´s Regular Simplex')
        label_explanation_4.config(font=('Arial', 14))
        canvas_gui.create_window(200, 33, window=label_explanation_4)

        # Label explanation 5
        label_explanation_5 = Label(root, text='2. Please choose a function and enter a initial guess for x/y:')
        label_explanation_5.config(font=('Arial', 10), bg='lavender')
        canvas_gui.create_window(200, 80, window=label_explanation_5)

        # Label Function
        label_function = Label(root, text='Function:')
        label_function.config(font=('Arial', 10), bg='white')
        canvas_gui.create_window(125, 130, window=label_function)

        # Function Dropdown - Menu
        function_Menu = ttk.Combobox(root, width=11, values=['Himmelblau', 'Rosenbrock'])
        function_Menu.current(0)
        canvas_gui.create_window(218, 130, window=function_Menu)

        # Label x/ y
        label_x = Label(root, text='x - value:')
        label_x.config(font=('Arial', 10), bg='white')
        canvas_gui.create_window(126, 190, window=label_x)

        label_y = Label(root, text='y - value:')
        label_y.config(font=('Arial', 10), bg='white')
        canvas_gui.create_window(126, 220, window=label_y)

        # Entry x/y
        entry_x = Entry(root, textvariable='x_value')
        entry_x.config(bg='white')
        canvas_gui.create_window(236, 190, window=entry_x)

        entry_y = Entry(root, textvariable='y_value')
        entry_y.config(bg='white')
        canvas_gui.create_window(236, 220, window=entry_y)

        # Optimize Button
        button_optimize = Button(root, font=('Arial', 13), command=optimize_clicked, text='Optimize!', bg='lightcoral')
        canvas_gui.create_window(200, 275, window=button_optimize)

        root.mainloop()

    elif initial_selection == 'Random Bounds':

        # optimize_clicked function (is called when the user clicks on the "Optimize!" Button)
        def optimize_clicked():
            global check_optimize_clicked
            check_optimize_clicked = True
            global check_float
            global selection
            global x
            global x0

            selection = function_Menu.get()  # selection of a optimization function
            try:
                x0[0][0] = float(entry_lb_x.get())  # x start value
                x0[0][1] = float(entry_lb_y.get())  # y start value
                x0[1][0] = float(entry_ub_x.get())  # x start value
                x0[1][1] = float(entry_ub_y.get())  # y start value
                check_float = True
            except:
                messagebox.showerror("Type Error", "The x/y - values must be of the type float. Please try again.")
                check_float = None
                check_optimize_clicked = None
            if check_float:
                x = random_bounds([x0[0][0], x0[0][1]], [x0[1][0], x0[1][1]])  # initialization of the simplex
                root.destroy()  # automatically closes the window if the algorithm finishes successfully
                init.destroy()

        # Root & UI for Random Bounds
        root = Tk()
        root.title('Nelder Mead Visualizer')
        root.resizable(0, 0)

        # Canvas
        canvas_gui = Canvas(root, width=500, height=400)
        canvas_gui.pack()

        # Label explanation 4
        label_explanation_4 = Label(root, text='Your choice: Random Bounds')
        label_explanation_4.config(font=('Arial', 14))
        canvas_gui.create_window(250, 33, window=label_explanation_4)

        # Label explanation 5
        label_explanation_5 = Label(root,
                                    text='2. Please choose a function and enter the lower and upper bounds for x/y:')
        label_explanation_5.config(font=('Arial', 10), bg='lavender')
        canvas_gui.create_window(250, 80, window=label_explanation_5)

        # Label Function
        label_function = Label(root, text='Function:')
        label_function.config(font=('Arial', 10), bg='white')
        canvas_gui.create_window(90, 130, window=label_function)

        # Function Dropdown - Menu
        function_Menu = ttk.Combobox(root, width=11, values=['Himmelblau', 'Rosenbrock'])
        function_Menu.current(0)
        canvas_gui.create_window(233, 130, window=function_Menu)

        # Label Lower Bound x/y
        label_lb_x = Label(root, text='Lower Bound x:')
        label_lb_x.config(font=('Arial', 10), bg='white')
        canvas_gui.create_window(110, 190, window=label_lb_x)

        label_lb_y = Label(root, text='Lower Bound y:')
        label_lb_y.config(font=('Arial', 10), bg='white')
        canvas_gui.create_window(110, 280, window=label_lb_y)

        # Label Upper Bound x/y
        label_ub_x = Label(root, text='Upper Bound x:')
        label_ub_x.config(font=('Arial', 10), bg='white')
        canvas_gui.create_window(110, 220, window=label_ub_x)

        label_ub_y = Label(root, text='Upper Bound y:')
        label_ub_y.config(font=('Arial', 10), bg='white')
        canvas_gui.create_window(110, 310, window=label_ub_y)

        # Entry Lower Bound x/y
        entry_lb_x = Entry(root, textvariable='lb_x_value')
        entry_lb_x.config(bg='white')
        canvas_gui.create_window(250, 190, window=entry_lb_x)

        entry_lb_y = Entry(root, textvariable='lb_y_value')
        entry_lb_y.config(bg='white')
        canvas_gui.create_window(250, 280, window=entry_lb_y)

        # Entry Upper Bound x/y
        entry_ub_x = Entry(root, textvariable='ub_x_value')
        entry_ub_x.config(bg='white')
        canvas_gui.create_window(250, 220, window=entry_ub_x)

        entry_ub_y = Entry(root, textvariable='ub_y_value')
        entry_ub_y.config(bg='white')
        canvas_gui.create_window(250, 310, window=entry_ub_y)

        # Optimize Button
        button_optimize = Button(root, font=('Arial', 13), command=optimize_clicked, text='Optimize!', bg='lightcoral')
        canvas_gui.create_window(250, 362, window=button_optimize)

        root.mainloop()


# GUI to choose initial simplex function
init = Tk()
init.title('Nelder Mead Visualizer')
init.resizable(0, 0)

# Canvas for initial simplex
canvas_init = Canvas(init, width=750, height=300)
canvas_init.pack()

# Label Title
label_title = Label(init, text='Visualization of the Nelder Mead Algorithm')
label_title.config(font=('Arial', 20))
canvas_init.create_window(375, 40, window=label_title)

# Label Explanation
label_explanation_1 = Label(init,
                            text='This program applies the Nelder Mead Algorithm to a chosen objective '
                                 'function and visualizes the results:')
label_explanation_1.config(font=('Arial', 10))
canvas_init.create_window(375, 85, window=label_explanation_1)

label_explanation_2 = Label(init, text='Step 1: Choose a method to determine the initial simplex.\n')
label_explanation_2.config(font=('Arial', 10))
canvas_init.create_window(375, 130, window=label_explanation_2)

label_explanation_3 = Label(init, text='  Step 2: Enter the needed parameters and select a function.')
label_explanation_3.config(font=('Arial', 10))
canvas_init.create_window(375, 141.5, window=label_explanation_3)

# Label Initial Simplex
label_init = Label(init, bg='lavender', text='1. Please choose one option for simplex-initialization: ')
label_init.config(font=('Arial', 10))
canvas_init.create_window(229, 180, window=label_init)

# Dropdown Menu for Function
initial_function = ttk.Combobox(init, width=24, values=['Spendley´s Regular Simplex', 'Random Bounds'])
initial_function.current(0)
canvas_init.create_window(490, 180, window=initial_function)

# Button Choose
button_choose = Button(init, font=('Arial', 13), command=function_chosen, text='Choose!', bg='lightcoral')
canvas_init.create_window(375, 240, window=button_choose)

init.mainloop()

"""
Algorithm Implementation================================================================================================
"""


# Optimization Function
def f(v):
    if selection == "Himmelblau":
        z = math.pow(math.pow(v[0], 2) + v[1] - 11, 2) + math.pow(v[0] + math.pow(v[1], 2) - 7,
                                                                  2)  # Himmelblau-Function
    else:
        z = (1 - v[0]) ** 2 + 100 * ((v[1] - (v[0] ** 2)) ** 2)  # Rosenbrock-Function

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
def centre(x):
    m = [0, 0]
    for i in [0, 1, 2]:
        m[0] += x[i][0]
        m[1] += x[i][1]
    m[0] = 1 / (len(x) - 1) * m[0]
    m[1] = 1 / (len(x) - 1) * m[1]
    return m


# Iteration of the algorithm
def iteration(m, x):
    alpha = 1
    beta =0.5
    gamma = 2
    delta = 0.5
    r = [0, 0]
    c = [0, 0]
    e = [0, 0]
    r[0] = m[0] + alpha * (m[0] - x[3][0])  # reflexion
    r[1] = m[1] + alpha * (m[1] - x[3][1])
    if f(r) < f(x[0]):
        e[0] = m[0] + gamma * (m[0] - x[3][0])  # expansion
        e[1] = m[1] + gamma * (m[1] - x[3][1])
        if f(e) < f(r):
            x[3] = e
        else:
            x[3] = r
    else:
        if f(r) < f(x[2]):
            x[3] = r
        else:
            c[0] = m[0] + beta * (m[0] - x[3][0])  # contraction
            c[1] = m[1] + beta * (m[1] - x[3][1])
            if f(c) < f(x[3]):
                x[3] = c
            else:
                for i in [1, 2, 3]:
                    x[i][0] = x[0][0] + delta * (x[i][0] - x[0][0])  # Shrink
                    x[i][1] = x[0][1] + delta * (x[i][1] - x[0][1])



"""
Plotly-Visualization Implementation=====================================================================================
"""


def visualize():
    # parameters/variables
    data = []
    x1 = []
    x2 = []
    x3 = []
    func = []
    button_position_x = 1.414  # define default button position
    button_position_y = 0.67
    rangex = [-5, 5]  # define default range of axis
    rangey = [-5, 5]
    rangez = [0, 300]
    number_of_traces = 7  # define default number of ranges
    # load all values of the csv file into one array
    with open('C:/Users/Andreas Schmid/Desktop/Projektstudium DataScience/nelder_mead.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append(row)

    #divide the values of the data-array into seperate x1, x2 and x3 arrays
    i = 0
    data_length = len(data)
    while i < data_length:
        x1.append(float(data[i][0]))
        x2.append(float(data[i][1]))
        x3.append(float(data[i][2]))
        func.append(data[i][4])
        i += 1

    # x, y, z values of the function
    x = np.outer(np.linspace(-4, 4, 30), np.ones(30))
    y = x.copy().T  # transpose
    if func[-1] == 'Himmelblau':
        z = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    elif func[-1] == 'Rosenbrock':
        z = (1 - x) ** 2 + 100 * ((y - (x ** 2)) ** 2)

    # initialize the figure for the visualization
    trace = go.Surface(x=x, y=y, z=z, opacity=0.2, cmax=1, cmin=0, showscale=False, hidesurface=True)
    data = [trace]
    fig = go.Figure(data=data)

    # plot function and highlight minimum/minima with markers (depending on the selected function)
    if func[-1] == 'Himmelblau':
        fig.add_trace(
            go.Surface(name='Himmelblau',  # define/design plot
                       x=x,
                       y=y,
                       z=z,
                       opacity=0.2,
                       cmax=1,
                       cmin=0,
                       showscale=False,
                       showlegend=False,
                       contours=dict(
                           y_show=True, x_show=True,
                           x_size=0.4, y_size=0.4,
                           y_color='silver',
                           x_color='silver',
                           y_start=-5, y_end=5,
                           x_start=-5, x_end=5)))
        fig.add_trace(
            go.Scatter3d(  # define/design minimum
                name='minimum 4: (3.58, -1.85)',
                mode='markers',
                x=[3.584428],
                y=[-1.848126],
                z=[0],
                marker=dict(
                    color='blue',
                    size=5,
                    opacity=0.6),
                showlegend=True
            ))
        fig.add_trace(
            go.Scatter3d(  # define/design minimum
                name='minimum 3: (-3.78, -3.28)',
                mode='markers',
                x=[-3.779319],
                y=[-3.283186],
                z=[0],
                marker=dict(
                    color='saddlebrown',
                    size=5,
                    opacity=0.6),
                showlegend=True
            ))
        fig.add_trace(
            go.Scatter3d(
                name='minimum 2: (-2.81, 3.13)',  # define/design minimum
                mode='markers',
                x=[-2.805118],
                y=[3.131312],
                z=[0],
                marker=dict(
                    color='tomato',
                    size=5,
                    opacity=0.6),
                showlegend=True
            ))
        fig.add_trace(
            go.Scatter3d(  # define/design minimum
                name='minimum 1: (3.00, 2.00)',
                mode='markers',
                x=[3],
                y=[2],
                z=[0],
                marker=dict(
                    color='lime',
                    size=5,
                    opacity=0.6),
                showlegend=True
            ))
        fig.add_trace(
            go.Scatter3d(  # dummy object for legend
                name='$f(x,y)=(x^2+y−11)^2+(x+y^2−7)^2$',
                mode='markers',
                x=[30],
                y=[30],
                z=[500],
                marker=dict(
                    color='yellow',
                    size=5,
                    opacity=0.8),
                showlegend=True,
                visible=True,
                opacity=1
            )
        )
    else:
        rangex = [-2, 2]  # change range of axis for rosenbrock visualization
        rangey = [-3.5, 4]
        rangez = [0, 1200]
        button_position_y = 0.755  # change button position for rosenbrock visualization
        number_of_traces = 4  # change number of traces for rosenbrock visualization
        fig.add_trace(
            go.Surface(name='Rosenbrock',  # define/design plot
                       x=x,
                       y=y,
                       z=z,
                       opacity=0.2,
                       cmax=1,
                       cmin=0,
                       contours=dict(
                           y_show=True, x_show=True,
                           x_size=0.4, y_size=0.4,
                           y_color='silver',
                           x_color='silver',
                           y_start=-3.5, y_end=4,
                           x_start=-2, x_end=2),
                       showscale=False,
                       showlegend=False))
        fig.add_trace(
            go.Scatter3d(  # define/design minimum
                name='minimum 1: (1.00, 1.00)',
                mode='markers',
                x=[1],
                y=[1],
                z=[0],
                marker=dict(
                    color='lime',
                    size=5,
                    opacity=0.6),
                showlegend=True
            )
        )
        fig.add_trace(
            go.Scatter3d(  # dummy object for legend
                name='$f(x,y)=(1-x^2)^2+100(y-x^2)^2$',
                mode='markers',
                x=[30],
                y=[30],
                z=[500],
                marker=dict(
                    color='yellow',
                    size=5,
                    opacity=0.8),
                showlegend=True,
                visible=True,
                opacity=1
            )
        )

    # add a scatter plot to the surface
    fig.add_scatter3d(
        x=[],
        y=[],
        z=[],
    )

    # update axis of the plot
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=rangex, autorange=False, nticks=8),
            yaxis=dict(range=rangey, autorange=False, nticks=8),
            zaxis=dict(range=rangez, autorange=False, nticks=8),
        )),

    # define/design the visualiztion of the result
    result = go.Scatter3d(
        name='result:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
             + '<span style="color:red">'
             + '(' + str(round(x1[-1], 2)) + ', ' + str(round(x2[-1], 2))
             + ')<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
             + '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Δ ≈ '
             + str(round(abs(x3[-1]), 2))
             + '</span>',
        mode='markers',
        x=[x1[-1], x1[-1]],
        y=[x2[-1], x2[-1]],
        z=[x3[-1], x3[-1]],
        marker=dict(
            color='red',
            size=5,
            symbol='x'),
        visible=True,
        showlegend=True,
        opacity=1
    )

    # 20 iteration with 4 points each
    iterations = range(0, data_length, 4)

    # iteration frames of the algorithm
    frames = [go.Frame(data=[go.Mesh3d(  # define/design tetrahedron
        x=x1[k:k + 4],
        y=x2[k:k + 4],
        z=x3[k:k + 4],
        colorscale=[[0, 'red'],
                    [0.5, 'springgreen'],
                    [1, 'white']],
        intensity=np.linspace(0, 1, 6, endpoint=True),
        intensitymode='cell',
        showscale=False)],

        traces=[number_of_traces],
        name=f'frame{k}'
    ) for k in iterations]

    frames.append(go.Frame(data=[result]))
    fig.update(frames=frames)

    # design complete layout
    fig.update_layout(
        title=dict(  # define/design title
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            text='Nelder Mead Algorithm - Function selected: '
                 + str(func[-1]),
            font=dict(size=21, color='black')),
        scene=dict(  # define/design axis and ticks
            xaxis=dict(tickfont=dict(color='black'),
                       titlefont=dict(color='black')),
            yaxis=dict(tickfont=dict(color='black'),
                       titlefont=dict(color='black')),
            zaxis=dict(tickfont=dict(color='black'),
                       titlefont=dict(color='black')),
            aspectmode='cube'),  # fix the size of the coordinate system to a cube
        margin_r=540,
        legend=dict(  # define/design legend
            title='<b>Legend:</b>',
            title_font_size=14,
            title_font_color='black',
            traceorder="reversed",
            bgcolor='aliceblue',
            bordercolor="black",
            borderwidth=0.5,
            font=dict(size=14, color='black'),
            itemsizing='constant',
            # itemwidth=60,
            y=0.95,
            x=1.1),
        updatemenus=[  # define/design play and pause buttons
            dict(
                type="buttons",
                buttons=[
                    dict(label="Pause ||",
                         method="animate",
                         args=[[None],
                               {"frame":
                                    {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition":
                                    {"duration": 0}}]),
                ],
                active=-1,
                showactive=False,
                font=dict(size=16, color='black'),
                bgcolor='white',
                bordercolor="black",
                borderwidth=1,
                y=button_position_y,
                x=button_position_x,
                direction='right'),

            dict(
                active=0,
                buttons=list([
                    dict(label="Play ▶ (fast)",
                         method="animate",
                         args=[None,
                               {"frame":
                                    {"duration": 100, "redraw": True},
                                "mode": "immediate",
                                "fromcurrent": True,
                                "transition":
                                    {"duration": 0,
                                     "easing":
                                         "linear"}}]),
                    dict(label="Play ▶ (medium)",
                         method="animate",
                         args=[None,
                               {"frame":
                                    {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "fromcurrent": True,
                                "transition":
                                    {"duration": 0,
                                     "easing":
                                         "linear"}}]),
                    dict(label="Play ▶ (slow)",
                         method="animate",
                         args=[None,
                               {"frame":
                                    {"duration": 500, "redraw": True},
                                "mode": "immediate",
                                "fromcurrent": True,
                                "transition":
                                    {"duration": 0,
                                     "easing":
                                         "linear"}}])

                ]),
                font=dict(size=15, color='black'),
                bgcolor='white',
                bordercolor="black",
                borderwidth=1,
                y=button_position_y,
                x=button_position_x - 0.117,
                direction='down'
            )

        ]

    )

    fig.show()


"""
main-function===========================================================================================================
"""


def main():
    global selection
    if check_float == None:
        return None

    header = ["x1", "x2", "f(x1, x2)", "Iteration", "Algorithmus"]  # header of the csv-sheet

    row1 = [x[0][0], x[0][1], f(x[0]), '0', selection]  # point 1
    row2 = [x[1][0], x[1][1], f(x[1]), '0', selection]  # point 2
    row3 = [x[2][0], x[2][1], f(x[2]), '0', selection]  # point 3
    row4 = [x[3][0], x[3][1], f(x[3]), '0', selection]  # point 4

    # write initial x, y, z values to csv sheets:
    with open('C:/Users/Andreas Schmid/Desktop/Projektstudium DataScience/nelder_mead.csv', 'w', newline="") as cs:
        write = csv.writer(cs)
        write.writerow(header)
        write.writerow(row1)
        write.writerow(row2)
        write.writerow(row3)
        write.writerow(row4)

    i = 1
    #while (f(x[3]) - f(x[0]))/(abs(f(x[3])) + abs(f(x[0])) + 1) < math.pow(10, -15):
    while i<200:
        x_sorted = sort(x)  # sort x-values according to function values
        centroid = centre(x_sorted)  # calculate centre
        iteration(centroid, x_sorted)  # iteration of the algorithm

        row1 = [x[0][0], x[0][1], f(x[0]), str(i), selection]
        row2 = [x[1][0], x[1][1], f(x[1]), str(i), selection]
        row3 = [x[2][0], x[2][1], f(x[2]), str(i), selection]
        row4 = [x[3][0], x[3][1], f(x[3]), str(i), selection]

        with open('C:/Users/Andreas Schmid/Desktop/Projektstudium DataScience/nelder_mead.csv', 'a', newline="") as cs:
            write = csv.writer(cs)
            write.writerow(row1)
            write.writerow(row2)
            write.writerow(row3)
            write.writerow(row4)

        i += 1
    if (check_optimize_clicked):  # visualization if the optimize button was clicked & all user entrys are correct
        visualize()


# call the main function
if __name__ == '__main__':
    main()
