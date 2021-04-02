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

opt_x1 = df._get_column_array(0)
#print(test[-1])

opt_x2 = df._get_column_array(1)

fig = px.scatter_3d(
    data_frame=df,
    x='x1',
    y='x2',
    z='f(x1, x2)',
    template='ggplot2',
    title='Nelder Mead Algorithm - Optimal values: (' + str(opt_x1[-1]) + ', ' + str(opt_x2[-1]) + ')',
    #text='Optimal x1-value: ' + opt_x1,
    height=700,
    animation_frame='Iteration',
    range_x=[min_x1-0.5, max_x1+0.5],
    range_y=[min_x2-0.5, max_x2+0.5],
    range_z=[min_z-0.5, max_z+0.5]
)

fig.update_traces(marker=dict(size=5,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig.update_layout(scene=dict(
                    xaxis=dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",),
                    yaxis=dict(
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    zaxis=dict(
                        backgroundcolor="rgb(230, 230,200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),),)

# fig.update_traces(marker_line=dict(), selector=dict(type='scattercarpet'))


pio.show(fig)
