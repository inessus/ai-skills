##  在线使用
```python
import plotly
#设置用户名和API-Key
plotly.tools.set_credentials_file(username='DemoAccount', api_key='lr1c37zw81')
```

##  离线使用
```python
import plotly.offline as py
from plotly.graph_objs import Scatter, Layout
py.init_notebook_mode(connected=True)
```

##  plot 和 iplot区别
py.plot会生成一个离线的html文件，里面放置图片。而py.iplot则直接在ipython notebook里面生成图片。


# 1 基本图形
## 1 Scatter 绘制
```python
import plotly
import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd
plotly.offline.init_notebook_mode(connected=True)

N = 1000
random_x = np.random.randn(N)
random_y = np.random.randn(N)

# create a trace
trace = go.Scatter(
    x = random_x,
    y = random_y,
    mode = 'markers'
)
data = [trace]

# plot and embed in ipython notebook
py.iplot(data)
```
## 2 line和Scatter
```py
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
py.init_notebook_mode(connected=True)

N = 100
random_x = np.linspace(0, 1, N)
random_y0 = np.random.randn(N) + 5
random_y1 = np.random.randn(N)
random_y2 = np.random.randn(N) - 5

# create traces
trace0 = go.Scatter(
    x = random_x,
    y = random_y0,
    mode = 'markers',
    name = 'markers'
)
trace1 = go.Scatter(
    x = random_x,
    y = random_y1,
    mode='lines+markers',
    name='lines+markers'
)
trace2 = go.Scatter(
    x = random_x,
    y = random_y2,
    mode = 'lines',
    name = 'lines'
)
data = [trace0, trace1, trace2]
py.iplot(data)
```

## 3 Scatter绘制风格
```python
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np

N = 500

trace0 = go.Scatter(
    x = np.random.randn(N),
    y = np.random.randn(N) + 2,
    name = 'above',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgb(152, 0, 0, .8)',
        line = dict(
            width=2,
            color='rgb(0, 0, 0)'
        )
    )
)
trace1 = go.Scatter(
    x = np.random.randn(N),
    y = np.random.randn(N) - 2,
    name='below',
    mode='markers',
    marker = dict(
        size = 10,
        color = 'rgb(255, 182, 193, .9)',
        line = dict(
            width = 2,
        )
    )
)
data = [trace0, trace1]
layout = dict(
    title = 'Styled Scatter',
    yaxis = dict(zeroline = False),
    xaxis = dict(zeroline = False)
)
fig = dict(data=data, layout=layout)
py.iplot(fig)
```

## 热敏感的数据标签
```python
import plotly.offline as py
import plotly.graph_objs as go
import random
import numpy as np
import pandas as pd
l = []
y = []
data= pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv")
N = 53
c = ['hsl('+str(h)+',50%'+',50%' for h in np.linspace(0, 360, N)]

for i in range(int(N)):
    y.append((2000+i))
    trace0 = go.Scatter(
        x = data['Rank'],
        y = data['Population'] + (i*1000000),
        mode = 'markers',
        marker = dict(
            size=14,
            line=dict(width=1),
            color=c[i],
            opacity=0.3),
        name = y[i],
        text=data['State'])
    l.append(trace0)

layout = go.Layout(
    title='Stats of USA States',
    hovermode='closest',
    xaxis=dict(
        title='Pop',
        ticklen=5,
        zeroline=False,
        gridwidth=2,
    ),
    yaxis=dict(
        title='Rank',
        ticklen=5,
        gridwidth=2,
    ),
    showlegend=False
)
fig = go.Figure(data=l, layout = layout)
py.iplot(fig)
```

## 4 Scatter颜色分布
```python
import plotly.graph_objs as go
import plotly.offline as py
import numpy as np

py.init_notebook_mode(connected=True)

trace1 = go.Scatter(
    y = np.random.randn(500),
    mode='markers',
    marker=dict(
        size='16',
        color=np.random.randn(500),
        colorscale='Viridis',
        showscale=True
    )
)
data = [trace1]
py.iplot(data)
```

## 5 大数据集
```python
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np

py.init_notebook_mode(connected=True)

N = 100000
trace = go.Scatter(
    x = np.random.randn(N),
    y = np.random.randn(N),
    mode = 'markers',
    marker = dict(
        color='#FFBAD2',
        line=dict(width=1)
    )
)
data = [trace]
py.iplot(data)
```

# line形绘制

## 1 简单线性绘制
```python
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np

py.init_notebook_mode(connected=True)

N = 500
random_x = np.linspace(0, 1, N)
random_y = np.random.randn(N)

trace = go.Scatter(
    x = random_x,
    y = random_y
)
data=[trace]
py.iplot(data)
```

## 2 线性绘制模式
```python
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np

N = 100
random_x = np.linspace(0, 1, N)
random_y0 = np.random.randn(N) + 5
random_y1 = np.random.randn(N)
random_y2 = np.random.randn(N) - 5

trace0 = go.Scatter(
    x = random_x,
    y = random_y0,
    mode = 'lines',
    name = 'lines'
)
trace1 = go.Scatter(
    x = random_x,
    y = random_y1,
    mode = 'lines+markers',
    name = 'lines+markers'
)
trace2 = go.Scatter(
    x = random_x,
    y = random_y2,
    mode = 'markers',
    name = 'markers'
)
data = [trace0, trace1, trace2]
py.iplot(data)
```

## 3 线性风格的绘制
```python

import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

month = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']

high_2000 = [32.5, 37.6, 49.9, 53.0, 69.1, 75.4, 76.5, 76.6, 70.7, 60.6, 45.1, 29.3]
low_2000 = [13.8, 22.3, 32.5, 37.2, 49.9, 56.1, 57.7, 58.3, 51.2, 42.8, 31.6, 15.9]
high_2007 = [36.5, 26.6, 43.6, 52.3, 71.5, 81.4, 80.5, 82.2, 76.0, 67.3, 46.1, 35.0]
low_2007 = [23.6, 14.0, 27.0, 36.8, 47.6, 57.7, 58.9, 61.2, 53.3, 48.5, 31.0, 23.6]
high_2014 = [28.8, 28.5, 37.0, 56.8, 69.7, 79.7, 78.5, 77.8, 74.1, 62.6, 45.3, 39.9]
low_2014 = [12.7, 14.3, 18.6, 35.5, 49.9, 58.0, 60.0, 58.6, 51.7, 45.2, 32.2, 29.1]

trace0 = go.Scatter(
    x = month,
    y = high_2014,
    name = 'High 2014',
    line = dict(
        color=('rgb(205, 12, 24)'),
        width = 4
    )
)

trace1 = go.Scatter(
    x = month,
    y = low_2014,
    name = 'Low 2014',
    line = dict(
        color=('rgb(22, 96, 167)'),
        width=4,
    )
)

trace2 = go.Scatter(
    x = month,
    y = high_2007,
    name = 'High 2007',
    line = dict(
        color=('rgb(205, 12, 24)'),
        width=4,
        dash='dash'
    )
)

trace3 = go.Scatter(
    x = month,
    y = low_2007,
    name = 'Low 2007',
    line = dict(
        color=('rgb(22, 96, 167)'),
        width = 4,
        dash = 'dash'
    )
)

trace4 = go.Scatter(
    x = month,
    y = high_2000,
    name = 'High 2007',
    line = dict(
        color=('rgb(205, 12, 24)'),
        width=4,
        dash='dot'
    )
)
trace5 = go.Scatter(
    x = month,
    y = low_2000,
    name = 'Low 2000',
    line = dict(
        color=('rgb(22, 96, 167)'),
        width=4,
        dash='dot'
    )
)
data = [trace0, trace1, trace2, trace3, trace4, trace5]

layout = dict(title='Averange High and low Temprarures in NewYork',
              xaxis = dict(title='Month'),
              yaxis = dict(title='Temperature (degrees F)'),
             )
fig = dict(data=data, layout=layout)
py.iplot(fig)
```

## 4 缺失数据处理
```python
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

trace1 = go.Scatter(
    x=[1, 2, 3, 4, 5,
       6, 7, 8, 9, 10,
       11, 12, 13, 14, 15],
    y=[10, 20, None, 15, 10,
       5, 15, None, 20, 10,
       10, 15, 25, 20, 10],
    name = '<b>No</b> Gaps', # Style name/legend entry with html tags
    connectgaps=True
)
trace2 = go.Scatter(
    x=[1, 2, 3, 4, 5,
       6, 7, 8, 9, 10,
       11, 12, 13, 14, 15],
    y=[5, 15, None, 10, 5,
       0, 10, None, 15, 5,
       5, 10, 20, 15, 5],
    name = 'Gaps',
)
data = [trace1, trace2]
fig = dict(data=data)
py.iplot(fig)
```

## 5 线性插值
```python
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

trace1 = go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[1, 3, 2, 3, 1],
    mode='lines+markers',
    name="'linear'",
    hoverinfo='name',
    line=dict(
        shape='linear'
    )
)
trace2 = go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[6, 8, 7, 8, 6],
    mode='lines+markers',
    name="'spline'",
    text=["tweak line smoothness<br>with 'smoothing' in line object"],
    hoverinfo='text+name',
    line=dict(
        shape='spline'
    )
)
trace3 = go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[11, 13, 12, 13, 11],
    mode='lines+markers',
    name="'vhv'",
    hoverinfo='name',
    line=dict(
        shape='vhv'
    )
)
trace4 = go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[16, 18, 17, 18, 16],
    mode='lines+markers',
    name="'hvh'",
    hoverinfo='name',
    line=dict(
        shape='hvh'
    )
)
trace5 = go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[21, 23, 22, 23, 21],
    mode='lines+markers',
    name="'vh'",
    hoverinfo='name',
    line=dict(
        shape='vh'
    )
)
trace6 = go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[26, 28, 27, 28, 26],
    mode='lines+markers',
    name="'hv'",
    hoverinfo='name',
    line=dict(
        shape='hv'
    )
)
data = [trace1, trace2, trace3, trace4, trace5, trace6]
layout = dict(
    legend=dict(
        y=0.5,
        traceorder='reversed',
        font=dict(
            size=16
        )
    )
)
fig = dict(data=data, layout=layout)
py.iplot(fig)
```

## 6 标注线性
```python
import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

title = 'Main Source for News'

labels = ['Television', 'Newspaper', 'Internet', 'Radio']

colors = ['rgba(67,67,67,1)', 'rgba(115,115,115,1)', 'rgba(49,130,189, 1)', 'rgba(189,189,189,1)']

mode_size = [8, 8, 12, 8]

line_size = [2, 2, 4, 2]

x_data = [
    [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2013],
    [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2013],
    [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2013],
    [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2013],
]

y_data = [
    [74, 82, 80, 74, 73, 72, 74, 70, 70, 66, 66, 69],
    [45, 42, 50, 46, 36, 36, 34, 35, 32, 31, 31, 28],
    [13, 14, 20, 24, 20, 24, 24, 40, 35, 41, 43, 50],
    [18, 21, 18, 21, 16, 14, 13, 18, 17, 16, 19, 23],
]

traces = []

for i in range(0, 4):
    traces.append(go.Scatter(
        x=x_data[i],
        y=y_data[i],
        mode='lines',
        line=dict(color=colors[i], width=line_size[i]),
        connectgaps=True,
    ))

    traces.append(go.Scatter(
        x=[x_data[i][0], x_data[i][11]],
        y=[y_data[i][0], y_data[i][11]],
        mode='markers',
        marker=dict(color=colors[i], size=mode_size[i])
    ))

layout = go.Layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        autotick=False,
        ticks='outside',
        tickcolor='rgb(204, 204, 204)',
        tickwidth=2,
        ticklen=5,
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=False,
    ),
    autosize=False,
    margin=dict(
        autoexpand=False,
        l=100,
        r=20,
        t=110,
    ),
    showlegend=False,
)

annotations = []

# Adding labels
for y_trace, label, color in zip(y_data, labels, colors):
    # labeling the left_side of the plot
    annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],
                                  xanchor='right', yanchor='middle',
                                  text=label + ' {}%'.format(y_trace[0]),
                                  font=dict(family='Arial',
                                            size=16,
                                            color=colors,),
                                  showarrow=False))
    # labeling the right_side of the plot
    annotations.append(dict(xref='paper', x=0.95, y=y_trace[11],
                                  xanchor='left', yanchor='middle',
                                  text='{}%'.format(y_trace[11]),
                                  font=dict(family='Arial',
                                            size=16,
                                            color=colors,),
                                  showarrow=False))
# Title
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Main Source for News',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
# Source
annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,
                              xanchor='center', yanchor='top',
                              text='Source: PewResearch Center & ' +
                                   'Storytelling with data',
                              font=dict(family='Arial',
                                        size=12,
                                        color='rgb(150,150,150)'),
                              showarrow=False))

layout['annotations'] = annotations

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)
```

## 7 填充线性

```
import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_rev = x[::-1]

# Line 1
y1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y1_upper = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y1_lower = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y1_lower = y1_lower[::-1]

# Line 2
y2 = [5, 2.5, 5, 7.5, 5, 2.5, 7.5, 4.5, 5.5, 5]
y2_upper = [5.5, 3, 5.5, 8, 6, 3, 8, 5, 6, 5.5]
y2_lower = [4.5, 2, 4.4, 7, 4, 2, 7, 4, 5, 4.75]
y2_lower = y2_lower[::-1]

# Line 3
y3 = [10, 8, 6, 4, 2, 0, 2, 4, 2, 0]
y3_upper = [11, 9, 7, 5, 3, 1, 3, 5, 3, 1]
y3_lower = [9, 7, 5, 3, 1, -.5, 1, 3, 1, -1]
y3_lower = y3_lower[::-1]

trace1 = go.Scatter(
    x=x+x_rev,
    y=y1_upper+y1_lower,
    fill='tozerox',
    fillcolor='rgba(0,100,80,0.2)',
    line=dict(color='transparent'),
    showlegend=False,
    name='Fair',
)
trace2 = go.Scatter(
    x=x+x_rev,
    y=y2_upper+y2_lower,
    fill='tozerox',
    fillcolor='rgba(0,176,246,0.2)',
    line=dict(color='transparent'),
    name='Premium',
    showlegend=False,
)
trace3 = go.Scatter(
    x=x+x_rev,
    y=y3_upper+y3_lower,
    fill='tozerox',
    fillcolor='rgba(231,107,243,0.2)',
    line=dict(color='transparent'),
    showlegend=False,
    name='Fair',
)
trace4 = go.Scatter(
    x=x,
    y=y1,
    line=dict(color='rgb(0,100,80)'),
    mode='lines',
    name='Fair',
)
trace5 = go.Scatter(
    x=x,
    y=y2,
    line=dict(color='rgb(0,176,246)'),
    mode='lines',
    name='Premium',
)
trace6 = go.Scatter(
    x=x,
    y=y3,
    line=dict(color='rgb(231,107,243)'),
    mode='lines',
    name='Ideal',
)

data = [trace1, trace2, trace3, trace4, trace5, trace6]

layout = go.Layout(
    paper_bgcolor='rgb(255,255,255)',
    plot_bgcolor='rgb(229,229,229)',
    xaxis=dict(
        gridcolor='rgb(255,255,255)',
        range=[1,10],
        showgrid=True,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        zeroline=False
    ),
    yaxis=dict(
        gridcolor='rgb(255,255,255)',
        showgrid=True,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        zeroline=False
    ),
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```

# Bar 图标绘制

## 简单的图形
```python
import plotly.offline as py
import plotly.graph_objs as go

data = [go.Bar(
        x=['giraffes', 'orangutans', 'monkeys'],
        y=[20, 14, 23]
)]
py.iplot(data)
```

## 分组bar图形
```python
import plotly.offline as py
import plotly.graph_objs as go

trace1 = go.Bar(
    x=['giraffes', 'orangutans', 'monkeys'],
    y=[20, 14, 23],
    name='SF Zoo'
)
trace2 = go.Bar(
    x=['giraffes', 'orangutans', 'monkeys'],
    y=[12, 18, 29],
    name='LA Zoo'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
```

## 堆叠bar图形
```python
import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

trace1 = go.Bar(
    x=['giraffes', 'orangutans', 'monkeys'],
    y=[20, 14, 23],
    name='SF Zoo'
)
trace2 = go.Bar(
    x=['giraffes', 'orangutans', 'monkeys'],
    y=[12, 18, 29],
    name='LA Zoo'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')
```

## 热敏感Bar图
```python
import plotly.offline as py
import plotly.graph_objs as go

trace0 = go.Bar(
    x=['Product A', 'Product B', 'Product C'],
    y=[20, 14, 23],
    text=['27% market share', '24% market share', '19% market share'],
    marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),
    opacity=0.6
)

data = [trace0]
layout = go.Layout(
    title='January 2013 Sales Report',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='text-hover-bar')
```

## 直接标签bar
```python
import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

x = ['Product A', 'Product B', 'Product C']
y = [20, 14, 23]

data = [go.Bar(
            x=x,
            y=y,
            text=y,
            textposition = 'auto',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )]

py.iplot(data, filename='bar-direct-labels')
```