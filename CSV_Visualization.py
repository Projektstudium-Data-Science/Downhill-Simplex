import pandas as pd
import plotly.express as px
import plotly.io as pio

x_eye = -1.25
y_eye = 2
z_eye = 0.5

df = pd.read_csv('nelder_mead.csv')

min_x1 = min(df.get('x1'))
max_x1 = max(df.get('x1'))
min_x2 = min(df.get('x2'))
max_x2 = max(df.get('x2'))
min_z = min(df.get('f(x1, x2)'))
max_z = max(df.get('f(x1, x2)'))

fig = px.scatter_3d(
    data_frame=df,
    x='x1',
    y='x2',
    z='f(x1, x2)',
    template='ggplot2',
    title='Nelder Mead Algorithm - Visualization',
    height=700,
    animation_frame='Iteration',
    range_x=[min_x1, max_x1],
    range_y=[min_x2, max_x2],
    range_z=[min_z, max_z]
)

pio.show(fig)
