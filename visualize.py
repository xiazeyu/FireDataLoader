import geopandas as gpd
import numpy as np
import plotly.express as px


def visualize_fire(fire_data: dict):
    
    data_dict = {
        'timestamp': fire_data.get('timestamps'),
        'geometry': fire_data.get('data')
    }

    gdf = gpd.GeoDataFrame(data_dict, geometry='geometry')

    gdf = gdf.sort_values('timestamp')
    gdf['Time'] = gdf['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    gdf['Status'] = 'Active Fire'

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    
    max_bound_diff = max(maxx - minx, maxy - miny)
    zoom_level = 9.5 - np.log(max_bound_diff) if max_bound_diff > 0 else 10

    fig = px.choropleth_map(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color='Status',
        animation_frame='Time',
        color_discrete_map={'Active Fire': '#FF4500'},
        opacity=0.6,
        center={"lat": center_lat, "lon": center_lon},
        zoom=zoom_level,
        title=f"Fire Progression: {fire_data.get('metadata').get('name')}"
    )
    
    fig.update_layout(
        map_style="white-bg",
        map=dict(
            center={"lat": center_lat, "lon": center_lon},
            zoom=zoom_level,
            layers=[
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "sourceattribution": "Esri, Maxar, Earthstar Geographics",
                    "source": [
                        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    ]
                }
            ]
        ),
        height=800, 
        margin={"r":0,"t":40,"l":0,"b":0},
        transition={'duration': 0}
    )

    fig.show()
