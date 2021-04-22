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
trace = go.Surface(x=x, y=y, z=z, opacity=0.2, cmax=1, cmin=0, showscale=False, hidesurface=True)  # , opacity=0.2
data = [trace]
layout = go.Layout(title='3D Surface plot')
fig = go.Figure(data=data)

fig.add_trace(go.Surface(x=x, y=y, z=z,opacity=0.2, cmax=1, cmin=0, showscale=False,showlegend=False))

# highlight minimum/minima with a red marker
if func[-1] == 'Himmelblau':
    fig.add_trace(
        go.Scatter3d(
            name='actual minimum',
            mode='markers',
            x=[3, -2.805118, -3.779319, 3.584428],
            y=[2, 3.131312, -3.283186, -1.848126],
            z=[0, 0, 0, 0],
            marker=dict(
                color='lime',
                size=5,),
            showlegend=True
        )
    )
    fig.add_trace(
        go.Scatter3d(
            name='f(x,y)=(x^2+y−11)^2+(x+y^2−7)^2',
            mode='markers',
            x=[30],
            y=[30],
            z=[500],
            marker=dict(
                color='yellow',
                size=5),
            showlegend=True,
            visible=True,
            opacity=1
        )
    )
else:
    fig.add_trace(
        go.Scatter3d(
            name='actual minimum',
            mode='markers',
            x=[3, -2.805118, -3.779319, 3.584428],
            y=[2, 3.131312, -3.283186, -1.848126],
            z=[0, 0, 0, 0],
            marker=dict(
                color='lime',
                size=5 ),
            showlegend=True
        )
    )
    fig.add_trace(
        go.Scatter3d(
            name='f(x,y)=(1-x)^2+100(y-x^2)^2',
            mode='markers',
            x=[30],
            y=[30],
            z=[500],
            marker=dict(
                color='yellow',
                size=5),
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
        xaxis=dict(range=[-5, 5], autorange=False, nticks=8),
        yaxis=dict(range=[-5, 5], autorange=False, nticks=8),
        zaxis=dict(range=[0, 300], autorange=False, nticks=8),
    )),

# add title
fig.update_layout(title=dict(text='Nelder Mead Algorithm - Function selected: '+ str(func[-1]),
                             font=dict(size=21)))

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

    traces=[4],
    name=f'frame{k}'
) for k in iterations]

optimalpoint = go.Scatter3d(
    name='result: ≈ ('+ str(round(x1[-1],4)) + ', ' + str(round(x2[-1],4)) + ')',
    mode='markers',
    x=[x1[-1], x1[-1]],
    y=[x2[-1], x2[-1]],
    z=[x3[-1], x3[-1]],
    marker=dict(
                color='red',
                size=5),
    visible=True,
    showlegend=True,
    opacity=1
    )

frames.append(go.Frame(data=[optimalpoint])
    )



fig.update(frames=frames)

# add play button for animation of the algorithm
fig.update_layout(updatemenus=[
                               dict(type="buttons", buttons=[play(), pause()],
                                    font=dict(size=14),bgcolor='aliceblue',bordercolor="whitesmoke",borderwidth=3.5,y=0.77,x=1.38, direction='right')])

# fix the size of the coordinate system to a cube
fig.update_layout(scene_aspectmode='cube')

#design legend
fig.update_layout(
    legend=dict(
        title='Legend:',
        title_font_size=16,
        traceorder="reversed",
        bgcolor='aliceblue',
        bordercolor="whitesmoke",
        borderwidth=3.5,
        font=dict(size=14),
        itemsizing='constant',
        itemwidth=60
        ))

fig.update_layout(
    title={
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    legend={
        'y':0.95,
        'x':1.1,
    })

fig.update_layout(margin_r=540)


fig.show()