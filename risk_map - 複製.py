# risk_map.py (增强版 - 连续色带)
import folium
import geopandas as gpd
import branca.colormap as cm
import numpy as np

# 读取数据
pipe = gpd.read_file("data/pipe_rainwater.geojson")
inlets = gpd.read_file("data/inlets_density.geojson")
risk = gpd.read_file("data/risk_zones.geojson")

# 地图中心
m = folium.Map(location=[22.3193, 114.1694], zoom_start=11)

# ---- 1. 渠管 ----
folium.GeoJson(
    pipe,
    name="雨水渠管",
    style_function=lambda x: {'color': 'blue', 'weight': 1.5, 'opacity': 0.6}
).add_to(m)

# ---- 2. 渠盖密度（分级设色） ----
def inlet_style(feature):
    count = feature['properties']['count']
    if count <= 2:
        color = '#ffffcc'
    elif count <= 10:
        color = '#ffeda0'
    elif count <= 30:
        color = '#feb24c'
    elif count <= 60:
        color = '#fd8d3c'
    else:
        color = '#bd0026'
    return {'fillColor': color, 'color': '#333', 'weight': 0.5, 'fillOpacity': 0.5}

folium.GeoJson(
    inlets,
    name="渠蓋密度",
    style_function=inlet_style,
    tooltip=folium.GeoJsonTooltip(fields=['count'], aliases=['🕳️ 渠蓋數量: '])
).add_to(m)

# ---- 3. 风险网格（连续色带） ----
scores = risk['risk_score']
vmin, vmax = scores.min(), scores.max()
colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=vmin, vmax=vmax)
colormap.caption = '積水風險分數'

def risk_style(feature):
    score = feature['properties']['risk_score']
    if np.isnan(score):
        color = '#cccccc'
    else:
        color = colormap(score)
    return {
        'fillColor': color,
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7
    }

folium.GeoJson(
    risk,
    name="積水風險",
    style_function=risk_style,
    tooltip=folium.GeoJsonTooltip(fields=['risk_score'], aliases=['風險分數: '])
).add_to(m)

# 添加图例
colormap.add_to(m)

# 图层控制
folium.LayerControl().add_to(m)

# 保存
m.save("risk_map.html")
print("✅ 增強版風險地圖已儲存：risk_map.html")