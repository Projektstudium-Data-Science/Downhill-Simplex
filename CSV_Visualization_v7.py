import math
import numpy as np
import plotly.graph_objs as go
import csv
from plotly.subplots import make_subplots

# parameters/variables
data = []
x1 = []
x2 = []
x3 = []
func = []
button_position_x = 1.385                                               #define default button position
button_position_y = 0.67
rangex=[-5,5]                                                           #define default range of axis
rangey=[-5,5]
rangez=[0,300]
# put all values of the csv file into one array
with open('nelder_mead.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        data.append(row)

# divide the values of the data-array into seperate x1, x2 and x3 arrays
i = 0
while i <= (19*4):
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
    z = (1 - x)**2 + 100 * ((y - (x**2))**2)

#initialize the figure for the visualization
trace = go.Surface(x=x, y=y, z=z, opacity=0.2, cmax=1, cmin=0, showscale=False, hidesurface=True)
data = [trace]
layout = go.Layout(title='3D Surface plot')
fig = go.Figure(data=data)



# plot function and highlight minimum/minima with markers (depending on the selected function)
if func[-1] == 'Himmelblau':
    fig.add_trace(
        go.Surface(name='Himmelblau',                                       #define/design plot
                   x=x,
                   y=y,
                   z=z,
                   opacity=0.2,
                   cmax=1,
                   cmin=0,
                   showscale=False,
                   showlegend=False,
                   contours=dict(
                                 y_show=True,x_show=True,
                                 x_size=0.4,y_size=0.4,
                                 y_color='silver',
                                 x_color='silver',
                                 y_start=-5,y_end=5,
                                 x_start = -5, x_end = 5 )))
    fig.add_trace(
        go.Scatter3d(                                                       #define/design minimum
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
        go.Scatter3d(                                                       #define/design minimum
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
            name='minimum 2: (-2.81, 3.13)',                                #define/design minimum
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
        go.Scatter3d(                                                       #define/design minimum
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
        go.Scatter3d(                                                      #dummy object for legend
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
    rangex = [-2, 2]                                                       #change range of axis
    rangey = [-3.5,4]                                                      #for rosenbrock visualization
    rangez = [0, 2000]
    button_position_y=0.755                                                #change button position
    button_position_x=1.368                                                #for rosenbrock visualization
    fig.add_trace(
        go.Surface(name='Rosenbrock',                                      #define/design plot
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
        go.Scatter3d(                                                      #define/design minimum
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
        go.Scatter3d(                                                     #dummy object for legend
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
        + '('+str(round(x1[-1],2)) + ', ' + str(round(x2[-1],2))
        +')<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        +'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Δ ≈ '
        + str(round(abs(x3[-1]),2))
        +'</span>',
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
iterations = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76]

# iteration frames of the algorithm
frames = [go.Frame(data=[go.Mesh3d(                                             #define/design tetrahedron
    x=x1[k:k+4],
    y=x2[k:k+4],
    z=x3[k:k+4],
    colorscale=[[0, 'red'],
                [0.5, 'springgreen'],
                [1, 'white']],
    intensity=np.linspace(0, 1, 6, endpoint=True),
    intensitymode='cell',
    showscale=False)],

    traces=[7],
    name=f'frame{k}'
) for k in iterations]

frames.append(go.Frame(data=[result]))
fig.update(frames=frames)

#design complete layout
fig.update_layout(
    title=dict(                                                                  #define/design title
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top',
                text='Nelder Mead Algorithm - Function selected: '
                     + str(func[-1]),
                font=dict(size=21, color='black')),
    scene=dict(                                                                 #define/design axis and ticks
                xaxis=dict(tickfont=dict(color= 'black'),
                           titlefont=dict(color='black')),
                yaxis=dict(tickfont=dict(color= 'black'),
                           titlefont=dict(color='black')),
                zaxis=dict(tickfont=dict(color= 'black'),
                           titlefont=dict(color='black')),
                aspectmode='cube'),                                             #fix the size of the coordinate system to a cube
    margin_r=540,
    legend = dict(                                                              #define/design legend
                title='<b>Legend:</b>',
                title_font_size=14,
                title_font_color='black',
                traceorder="reversed",
                bgcolor='aliceblue',
                bordercolor="black",
                borderwidth=0.5,
                font=dict(size=14, color='black'),
                itemsizing='constant',
                itemwidth=60,
                y=0.95,
                x=1.1),
    updatemenus=[                                                               #define/design play and pause buttons
        dict(
                type="buttons",
                buttons=[
                    dict(label="Play  ▶",
                         method="animate",
                         args= [None,
                                {"frame":
                                    {"duration": 800, "redraw": True},
                                "mode": "immediate",
                                 "fromcurrent": True,
                                 "transition":
                                     {"duration": 0,
                                      "easing":
                                      "linear"}}]),
                    dict(label="Pause ||",
                         method="animate",
                         args= [[None],
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