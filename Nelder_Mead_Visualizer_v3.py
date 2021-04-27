import copy
import csv
import numpy as np
import math
import plotly.graph_objs as go
from tkinter import *
from tkinter import ttk, messagebox
from operator import length_hint

x = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

# GUI variables
check_float = None  # checks if a float-value is added to the input field
check_optimize_clicked=None # checks if the optimize-Button is clicked

"""
Initial Simplex for Nelder Mead ========================================================================================
"""


# Initialization with Random Bounds
def random_bounds(lb, ub):
    theta = [0.2, 0.4, 0.6, 0.8]
    for i in [0, 1, 2, 3]:
        for j in [0, 1]:
            x[i][j] = lb[j] + theta[i] * (ub[j] - lb[j])
        x[i][2] = 0
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
        x[i][2] = 0
    return x


"""
GUI Implementation =====================================================================================================
"""


# is called when the function for the initial simplex is chosen
def function_chosen():
    global initial_selection
    initial_selection = initial_function.get()  # selection of the function for the initial simplex

    if initial_selection == 'Spendley´s Regular Simplex':

        #  variables
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
                check_optimize_clicked=None
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
        label_explanation_5.config(font=('Arial', 10),bg='lavender')
        canvas_gui.create_window(200, 80, window=label_explanation_5)

        # Label Function
        label_function = Label(root, text='Function:')
        label_function.config(font=('Arial', 10), bg='white')
        canvas_gui.create_window(125, 130, window=label_function)

        # Function Dropdown - Menu
        function_Menu = ttk.Combobox(root,width=11, values=['Himmelblau', 'Rosenbrock'])
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
            check_optimize_clicked=True
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
        label_explanation_5 = Label(root, text='2. Please choose a function and enter the lower and upper bounds for x/y:')
        label_explanation_5.config(font=('Arial', 10), bg='lavender')
        canvas_gui.create_window(250, 80, window=label_explanation_5)

        # Label Function
        label_function = Label(root, text='Function:')
        label_function.config(font=('Arial', 10), bg='white')
        canvas_gui.create_window(90, 130, window=label_function)

        # Function Dropdown - Menu
        function_Menu = ttk.Combobox(root,width=11, values=['Himmelblau', 'Rosenbrock'])
        function_Menu.current(0)
        canvas_gui.create_window(233,130, window=function_Menu)

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

        #  Entry Upper Bound x/y
        entry_ub_x = Entry(root, textvariable='ub_x_value')
        entry_ub_x.config(bg='white')
        canvas_gui.create_window(250, 220, window=entry_ub_x)

        entry_ub_y = Entry(root, textvariable='ub_y_value')
        entry_ub_y.config(bg='white')
        canvas_gui.create_window(250, 310, window=entry_ub_y)

        #  Optimize Button
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

label_explanation_2 = Label(init,text='Step 1: Choose a method to determine the initial simplex.\n')
label_explanation_2.config(font=('Arial', 10))
canvas_init.create_window(375, 130, window=label_explanation_2)

label_explanation_3 = Label(init,text='  Step 2: Enter the needed parameters and select a function.')
label_explanation_3.config(font=('Arial', 10))
canvas_init.create_window(375, 141.5, window=label_explanation_3)

# Label Initial Simplex
label_init = Label(init,bg='lavender',text='1. Please choose one option for simplex-initialization: ')
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


'''
    Pure Python/Numpy implementation of the Nelder-Mead algorithm.
    Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
'''


def nelder_mead(f, x_start, no_improve_thr=10e-6, no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)
        return: tuple (best parameter array, best score)
    '''

    # init
    dim = 3
    prev_best = f(x_start[0])
    no_improv = 0
    res = []

    for i in [0, 1, 2, 3]:
        score = f(x_start[i])
        x_start[i] = np.array([x_start[i][0], x_start[i][1], 0.])
        res.append([x_start[i], score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]
        best_res = res

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        print('...best so far:', best, '...res:', res[0][0][1])
        row1 = [res[0][0][0], res[0][0][1], res[0][1], str(iters), selection]
        row2 = [res[1][0][0], res[1][0][1], res[1][1], str(iters), selection]
        row3 = [res[2][0][0], res[2][0][1], res[2][1], str(iters), selection]
        row4 = [res[3][0][0], res[3][0][1], res[3][1], str(iters), selection]

        with open('nelder_mead.csv', 'a') as cs:
            write = csv.writer(cs)
            write.writerow(row1)
            write.writerow(row2)
            write.writerow(row3)
            write.writerow(row4)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres


# Optimization Function
def f(v):
    if selection == "Himmelblau":
        z = (v[0]**2 + v[1] - 11)**2 + (v[0] + v[1]**2 - 7)**2  # Himmelblau-Funktion
    else:
        z = (1 - v[0])**2 + 100 * ((v[1] - (v[0]**2))**2)  # Rosenbrock-Funktion

    return z


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
    button_position_x = 1.385  # define default button position
    button_position_y = 0.67
    rangex = [-5, 5]  # define default range of axis
    rangey = [-5, 5]
    rangez = [0, 300]
    number_of_traces=7 # define default number of ranges
    # put all values of the csv file into one array
    with open('nelder_mead.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append(row)

    # divide the values of the data-array into seperate x1, x2 and x3 arrays
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
    y = x.copy().T  #  transpose
    if func[-1] == 'Himmelblau':
        z = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    elif func[-1] == 'Rosenbrock':
        z = (1 - x) ** 2 + 100 * ((y - (x ** 2)) ** 2)

    # initialize the figure for the visualization
    trace = go.Surface(x=x, y=y, z=z, opacity=0.2, cmax=1, cmin=0, showscale=False, hidesurface=True)
    data = [trace]
    layout = go.Layout(title='3D Surface plot')
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
        button_position_x = 1.368
        number_of_traces=4         # change number of traces for rosenbrock visualization
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
            size=3.2,
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
                    dict(label="Play  ▶",
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
                    dict(label="Pause ||",
                         method="animate",
                         args=[[None],
                               {"frame":
                                    {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition":
                                    {"duration": 0}}])
                ],
                active=-1,
                font=dict(size=16, color='black'),
                bgcolor='white',
                bordercolor="black",
                borderwidth=1,
                y=button_position_y,
                x=button_position_x,
                direction='right')])

    fig.show()


def main():
    global selection
    if check_float == None:
        return None
    import numpy as np

    header = ["x1", "x2", "f(x1, x2)", "Iteration", "Algorithmus"]  # header of the csv-sheet
    with open('nelder_mead.csv', 'w') as cs:
        write = csv.writer(cs)
        write.writerow(header)

    nelder_mead(f, x)


if __name__ == "__main__":
    main()
    if (check_optimize_clicked):  # visualization if the optimize button was clicked & all user entrys are correct
        visualize()


