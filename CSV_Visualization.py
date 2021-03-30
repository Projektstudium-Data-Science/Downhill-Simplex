import pandas as pd
import plotly.express as px
import plotly.io as pio

x_eye = -1.25
y_eye = 2
z_eye = 0.5

df = pd.read_csv('nelder_mead.csv')
# df = df[df['Iteration'].isin(['0'])]
# df = df[df['Iteration'].isin(['0', '1', '2', '3'])]

fig = px.scatter_3d(
    data_frame=df,
    x='x1',
    y='x2',
    z='f(x1, x2)',
    template='ggplot2',
    title='Nelder Mead Algorithm - Visualization',
    height=700,
    animation_frame='Iteration',
    range_x=[-3.5, -2.5],
    range_y=[2.5, 3.5],
    range_z=[0, 5]
)

pio.show(fig)
