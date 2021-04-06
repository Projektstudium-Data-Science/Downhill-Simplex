import pandas as pd
import plotly.express as px
import plotly.io as pio

df = pd.read_csv('nelder_mead.csv')     # Read CSV-File

# Define Range of x1, x2 and x3 values
min_x1 = min(df.get('x1'))
max_x1 = max(df.get('x1'))
min_x2 = min(df.get('x2'))
max_x2 = max(df.get('x2'))
min_z = min(df.get('f(x1, x2)'))
max_z = max(df.get('f(x1, x2)'))

# put values of the x1-, x2- and function-column into an array
opt_x1 = df._get_column_array(0)
opt_x2 = df._get_column_array(1)
func = df._get_column_array(4)

# plot the points in a 3d coordinate system
fig = px.scatter_3d(
    data_frame=df,
    x='x1',
    y='x2',
    z='f(x1, x2)',
    template='ggplot2',
    title='Nelder Mead Algorithm - Function ' + str(func[1]) + '\t\t\t\t\t\t Optimal values: (' + str(opt_x1[-1]) + ', '
          + str(opt_x2[-1]) + ')',    # the optimal value is the last value in the CSV-file (position [-1] in the array)
    animation_frame='Iteration',        # animate the algorithm according to the value in the 'Iteration'-column of the
                                        # CSV-file
    range_x=[min_x1-0.5, max_x1+0.5],   # range of the x1-axis
    range_y=[min_x2-0.5, max_x2+0.5],   # range of the x2-axis
    range_z=[min_z-0.5, max_z+0.5]      # range of the f(x1, x2)-axis
)

# set the size of the points to 5
fig.update_traces(marker=dict(size=5,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

# update color and depiction of the planes and the grid
fig.update_layout(scene=dict(
                    xaxis=dict(
                         nticks=8,
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",),
                    yaxis=dict(
                        nticks=8,
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    zaxis=dict(
                        nticks=8,
                        backgroundcolor="rgb(230, 230,200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),),)

# fix the size of the coordinate system to a cube
fig.update_layout(scene_aspectmode='cube')

# show the figure
pio.show(fig)
