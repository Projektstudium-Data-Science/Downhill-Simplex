import numpy as np
import plotly.graph_objs as go
import pandas as pd
import csv

# parameters
data = []
x1 = []
x2 = []
x3 = []

# Read CSV-File
df = pd.read_csv('nelder_mead.csv')     

# put values of the x1-, x2- and function-column into an array
opt_x1 = df._get_column_array(0)
opt_x2 = df._get_column_array(1)
opt_x3 = df._get_column_array(2)
func = df._get_column_array(4)

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
    i += 1

# x, y, z values of the function
x = np.outer(np.linspace(-4, 4, 30), np.ones(30))
y = x.copy().T  # transpose
if func[-1] == 'Himmelblau':
    z = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
elif func[-1] == 'Rosenbrock':
    z = (1 - x)**2 + 100 * ((y - (x**2))**2)

# plot a surface of the function
trace = go.Surface(x=x, y=y, z=z,cmax=1, opacity=0.2, cmin=0, showscale=False)
data = [trace]
layout = go.Layout(title='3D Surface plot')
fig = go.Figure(data=data)

# add a scatter plot to the surface
fig.add_scatter3d(
    x=[],
    y=[],
    z=[],
)

# update axis of the plot
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-5, 5], autorange=False, nticks=8),
        yaxis=dict(range=[-5, 5], autorange=False, nticks=8),
        zaxis=dict(range=[0, 300], autorange=False, nticks=8),
    )),

# add title and optimal value
fig.update_layout(title='Nelder Mead Algorithm - Function ' + str(func[1]) + '\t\t\t\t\t\t Optimal values: (' + str(opt_x1[-1]) + ', '
          + str(opt_x2[-1]) + ')',)

# 20 iteration with 4 points each
iterations = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76]

# iteration frames of the algorithm
frames = [go.Frame(data=[go.Mesh3d(
    x=x1[k:k+4],
    y=x2[k:k+4],
    z=x3[k:k+4],
    colorscale=[[0, 'red'],
                [0.5, 'mediumturquoise'],
                [1, 'magenta']],
    # Intensity of each vertex, which will be interpolated and color-coded
    intensity=np.linspace(0, 1, 4, endpoint=True),
    intensitymode='cell',
    showscale=False)],

    traces=[1],
    name=f'frame{k}'
) for k in iterations]
fig.update(frames=frames)

# add play button for animation of the algorithm
fig.update_layout(updatemenus=[dict(type="buttons",
                                    buttons=[dict(label="Play",
                                                  method="animate",
                                                  args=[None, dict(
                                                      frame=dict(redraw=True, fromcurrent=True, mode='immediate'))])])])

# add optimal endpoint
fig.add_scatter3d(
    x=[opt_x1[-1], opt_x1[-1]],
    y=[opt_x2[-1], opt_x2[-1]],
    z=[opt_x3[-1], opt_x3[-1]]
)

# fix the size of the coordinate system to a cube
fig.update_layout(scene_aspectmode='cube')

fig.show()

