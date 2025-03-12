# -*- coding: utf-8 -*-
"""
Oscar Bergstrom Dissertation Master Program
"""
import numpy as np
from numpy import sin, cos, pi
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, html, Output, callback, Input
from dash import dcc
import networkx as nx

pio.renderers.default = "browser"
# http://127.0.0.1:8050/
# For the Application

app = Dash(__name__)

app.layout = [
    html.Div("Oscar Bergstrom Dissertation Project"),
    html.Hr(),
    html.Div(id='click-data-output'),
    dcc.Graph(figure={}, id='star-paths', style={'width': '1200px',
                                                 'height': '1200px'}),
    dcc.Slider(
        id='hop-length-slider',
        min=0.1,
        max=20,
        step=0.1,
        value=2.0,
        marks={i: f"{i:.1f}" for i in np.arange(0.0, 20.0, 1)},
    ),
    dcc.Markdown()
]

data = pd.read_csv("new_distances.csv")

def data_frame_creation(data):
     
     df = data
     colour_selection = ['Red', 'Orange', 'Blue']
     colour = ['Orange']
     x = [0]
     y = [0]
     z = [0]
     sizes = [5]
     name = ['Sun']
     
     for i in range(0, len(df)):
        ra_rad = float(df.iloc[i, 2]) * pi / 180  # Right Ascension in radians
        dec_rad = float(df.iloc[i, 3]) * pi / 180  # Declination in radians
        distance = float(df.iloc[i, 4])  # Assuming 'Mas' is distance in parsecs (or convert accordingly)
   
        # 3D Cartesian coordinates based on spherical-to-Cartesian transformation
        x.append((1000 * distance) * 3.26 * sin(ra_rad) * cos(dec_rad))
        y.append((1000 * distance) * 3.26 * sin(ra_rad) * sin(dec_rad))
        z.append((1000 * distance) * 3.26 * cos(ra_rad))
        num = np.random.randint(0,2)
        colour.append(colour_selection[num])
        sizes.append(5)
        name.append(df.iloc[i,10])
            
     x = np.array(x)
     y = np.array(y)
     z = np.array(z)
     points = np.stack((x, y, z), axis = 1)
             
     return name,x,y,z,colour,sizes,points

name,x,y,z,colour,sizes,points = data_frame_creation(data)

@callback(
    [Output('star-paths', 'figure'), Output('click-data-output', 'data')],
    [Input('hop-length-slider', 'value'), Input('star-paths','clickData')]
    )



def plotting_process(hop_length, clickData):
    print(f"The hop length: {hop_length:.1f} light-years")
    
    fig = go.Figure()
   
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', hovertemplate=name, name="",
                                        marker=dict(size = sizes, 
                                                    color = colour)))
    paths = []
   
     ### Establishes NetorkX graph object
    G  = nx.Graph()
    
    for i in range(len(points)):
        for j in range(0, len(points)):
            distance = np.linalg.norm(points[i]-points[j])
            if hop_length >= distance:
                paths.append({
                    'from': points[i],
                    'to': points[j],
                    'distance': distance,
                    'x': [x[i], x[j]],
                    'y': [y[i], y[j]],
                    'z': [z[i], z[j]]
                }) #Chat Gpt helped here
                G.add_edge(tuple(points[i]), tuple(points[j]), weight = distance)
     
    if clickData is None or 'points' not in clickData:
         click_output = "No Star Selected"
    else:
         press = clickData['points'][0]
         x_data = press['x']
         y_data = press['y']
         z_data = press['z']
       
         click_output = (x_data, y_data, z_data)
         
         start_point = (0,0,0)
         if nx.has_path(G, start_point, click_output):
             shortest_path = nx.shortest_path(G, start_point, click_output, weight = 'weight')
         
             length = nx.shortest_path_length(G, start_point, click_output, weight = 'weight')
             x_dij, y_dij, z_dij = np.array(shortest_path).T ## Provided by Chatgpt
         
         
         ### Set speed
         
             travel_speed = 0.02 # In units of c
         
             travel_time = length/travel_speed
             print("The time taken to travel to your chosen star at speed {}c is: {:.2f} years".format(travel_speed, travel_time))
         
             fig.add_trace(go.Scatter3d(x=x_dij, y=y_dij, z=z_dij, name="", hovertemplate=None, 
                                        mode='lines', line=dict(color='red', width=5))) 
         else:
            print("No path found!")
                  
    for path in paths:
        fig.add_trace(go.Scatter3d(x=path['x'], y=path['y'], z=path['z'], name="", hovertemplate=None, 
                                   mode='lines', line=dict(color='gray', width=2)))        
 
    fig.update_layout(template="seaborn")
    """
    fig.update_layout(
        
         scene=dict(
             bgcolor='black',  # Background color of the 3D plot area
             xaxis=dict(
                 #color='black',  # Axis label color
                 #gridcolor='black'  # Grid line color (can be customized)
                 ),
             yaxis=dict(
                 #color='black',
                 #gridcolor='black'
                 ),
             zaxis=dict(
                 #color='black',
                 #gridcolor='black'
                 )
             ),
         paper_bgcolor='black',  # Background color of the entire figure
         font=dict(color='white')  # Font color for axes and text
         )
    """
    return fig, click_output    

if __name__ == "__main__":
    app.run(debug=True)

