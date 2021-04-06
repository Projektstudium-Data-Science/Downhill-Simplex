import pandas as pd
import plotly.express as px
import plotly.io as pio

df = pd.read_csv('nelder_mead.csv')

min_x1 = min(df.get('x1'))
max_x1 = max(df.get('x1'))
min_x2 = min(df.get('x2'))
max_x2 = max(df.get('x2'))
min_z = min(df.get('f(x1, x2)'))
max_z = max(df.get('f(x1, x2)'))

opt_x1 = df._get_column_array(0)
opt_x2 = df._get_column_array(1)
func = df._get_column_array(4)

fig = px.scatter_3d(
    data_frame=df,
    x='x1',
    y='x2',
    z='f(x1, x2)',
    template='ggplot2',
    title='Nelder Mead Algorithm - Function ' + str(func[1]) + '\t\t\t\t\t\t Optimal values: (' + str(opt_x1[-1]) + ', ' + str(opt_x2[-1]) + ')',
    animation_frame='Iteration',
    range_x=[min_x1-0.5, max_x1+0.5],
    range_y=[min_x2-0.5, max_x2+0.5],
    range_z=[min_z-0.5, max_z+0.5]
)

fig.update_traces(marker=dict(size=5,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
"""
fig.update_traces(projection_x=0.67, projection_y=0.67, projection_z=0.67,
                  selector=dict(mode='scatter3d'))
"""
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
# fig.update_traces(marker_line=dict(), selector=dict(type='scattercarpet'))

fig.update_layout(scene_aspectmode='cube')

pio.show(fig)
