import numpy as np
import plotly.graph_objs as go
import csv
from plotly.subplots import make_subplots


# define "play" and "pause" - Buttons
def play(frame_duration=800, transition_duration=0):
    return dict(label="Play", method="animate", args=
    [None, {"frame": {"duration": frame_duration, "redraw": True},
            "mode": "immediate",
            "fromcurrent": True, "transition": {"duration": transition_duration, "easing": "linear"}}])


def pause():
    return dict(label="Pause", method="animate", args=
    [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])


# parameters
data = []
x1 = []
x2 = []
x3 = []
func = []

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

# plot a surface of the function
trace = go.Surface(x=x, y=y, z=z, opacity=0.2, cmax=1, cmin=0, showscale=False)  # , opacity=0.2
data = [trace]
layout = go.Layout(title='3D Surface plot')
fig = go.Figure(data=data)

fig.add_scatter3d(
    x=[3, -2.805118, -3.779319, 3.584428],
    y=[2, 3.131312, -3.283186, -1.848126],
    z=[0, 0, 0, 0],
    mode='markers'
)

# surface with minima

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
fig.update_layout(title='Nelder Mead Algorithm - Function ' + str(func[-1]) + '\t\t\t\t\t\t Optimal values: (' +
                        str(x1[-1]) + ', ' + str(x2[-1]) + ')',)

# 20 iteration with 4 points each
iterations = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76]

# iteration frames of the algorithm
frames = [go.Frame(data=[go.Mesh3d(
    x=x1[k:k+4],
    y=x2[k:k+4],
    z=x3[k:k+4],
    colorscale=[[0, 'fuchsia'],
                [0.5, 'greenyellow'],
                [1, 'magenta']],
    # Intensity of each vertex, which will be interpolated and color-coded
    intensity=np.linspace(0, 1, 6, endpoint=True),
    intensitymode='cell',
    showscale=False)],

    traces=[1],
    name=f'frame{k}'
) for k in iterations]

frames.append(go.Frame(data=[go.Scatter3d(
    x=[x1[-1], x1[-1]],
    y=[x2[-1], x2[-1]],
    z=[x3[-1], x3[-1]])]))

frames.append(go.Frame(data=[go.Surface(x=x, y=y, z=z, opacity=0.2, cmax=1, cmin=0, showscale=False)]))

fig.update(frames=frames)

# add play button for animation of the algorithm
fig.update_layout(updatemenus=[dict(type="buttons", buttons=[play(), pause()])])

# fix the size of the coordinate system to a cube
fig.update_layout(scene_aspectmode='cube')

fig.show()
