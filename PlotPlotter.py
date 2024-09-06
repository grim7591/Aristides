import folium
import pandas as pd
from folium import plugins
import numpy as np

#MapData = MapData.copy()

def PlotPlotter(MapData,MarketClusterID):
    if MarketClusterID == '300':
        MapData_filtered = MapData[MapData['imprv_type_cd'] == MarketClusterID].copy()
    else: MapData_filtered = MapData[MapData['Market_Cluster_ID'] == MarketClusterID].copy()
    conditions = [
        (MapData_filtered['sale_ratio'] < 0.7), # condition 1
        (MapData_filtered['sale_ratio'] >= 0.7) & (MapData_filtered['sale_ratio'] <=1), # condition 2
        (MapData_filtered['sale_ratio'] >= 1)
    ]
    MapData_filtered['url'] = MapData_filtered['url'] = 'https://qpublic.schneidercorp.com/Application.aspx?AppID=1081&LayerID=26490&PageTypeID=4&PageID=10770&Q=336226735&KeyValue=' + MapData_filtered['geo_id'].astype(str)

    colors = 'blue', 'gray', 'red'

    m = folium.Map(zoom_start=12, 
                tiles="cartodb positron")

    MapData_filtered.loc[:,'color'] = np.select(conditions, colors, default='green')

    radius = 6
    for i in range(len(MapData_filtered)):
        # Create HTML content for the popup with a link
        popup_html = f'<a href="{MapData_filtered.iloc[i]["url"]}" target="_blank">Link to Site</a>'
        popup = folium.Popup(popup_html, max_width=2650)
        
        # Create a CircleMarker and add it to the map
        folium.CircleMarker(
            location=[MapData_filtered.iloc[i]['CENTROID_Y'], MapData_filtered.iloc[i]['CENTROID_X']],
            radius=radius,
            color=MapData_filtered.iloc[i]['color'],  # Removed square brackets around the color
            stroke=False,
            fill=True,
            fill_opacity=0.6,
            opacity=1,
            popup=popup
        ).add_to(m)

    try:
        filename = f"Outputs/{MarketClusterID}.html"
        print(f"Saving map to {filename}...")
        m.save(filename)
        print(f"Map saved to {filename}")
        print(f"Map saved to {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")