import folium
import pandas as pd
from folium import plugins
import numpy as np
MapData = MapData.copy()

MapData_filtered = MapData[MapData['Market_Cluster_ID'] == 'HighSprings_B']
conditions = [
    (MapData_filtered['sale_ratio'] < 0.75), # condition 1
    (MapData_filtered['sale_ratio'] = 0.75), # condition 2
]

m = folium.Map(zoom_start=12, 
               tiles="cartodb positron")

MapData_filtered['color'] = np.where(MapData_filtered['sale_ratio'] < 0.75, 'red', 'green')

radius = 6
for i in range(0,len(MapData_filtered)):
    
    folium.CircleMarker(
         location=[MapData_filtered.iloc[i]['CENTROID_Y'], MapData_filtered.iloc[i]['CENTROID_X']],
        radius=radius,
        color= [MapData_filtered.iloc[i]['color']],
        stroke=False,
        fill=True,
        fill_opacity=0.6,
        opacity=1
    ).add_to(m)

m