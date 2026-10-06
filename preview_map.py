# preview_map.py (終極保證版)
import folium
import geopandas as gpd

# 讀取數據
pipe = gpd.read_file("data/pipe_rainwater.geojson")
inlets = gpd.read_file("data/inlets_density.geojson")

print(f"✅ 渠管數量: {len(pipe)}")
print(f"✅ 網格數量: {len(inlets)}")

# 計算地圖中心（自動對準數據範圍）
bounds = inlets.total_bounds  # [minx, miny, maxx, maxy]
center_lat = (bounds[1] + bounds[3]) / 2
center_lng = (bounds[0] + bounds[2]) / 2

m = folium.Map(location=[center_lat, center_lng], zoom_start=13)

# --- 1. 渠管（藍色） ---
folium.GeoJson(
    pipe,
    name="雨水渠管",
    style_function=lambda x: {'color': 'blue', 'weight': 1.5, 'opacity': 0.6}
).add_to(m)

# --- 2. 進水渠蓋密度（鮮明顏色） ---
def style_function(feature):
    count = feature['properties']['count']
    # 顏色由淺黃到深紅，非常明顯
    if count <= 2:
        color = '#ffffcc'   # 極淺黃
    elif count <= 10:
        color = '#ffeda0'   # 淺黃
    elif count <= 30:
        color = '#feb24c'   # 橙色
    elif count <= 60:
        color = '#fd8d3c'   # 深橙
    else:
        color = '#bd0026'   # 深紅
    return {
        'fillColor': color,
        'color': '#333333',
        'weight': 0.5,
        'fillOpacity': 0.7
    }

folium.GeoJson(
    inlets,
    name="進水渠蓋密度",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(fields=['count'], aliases=['🕳️ 渠蓋數量: '], localize=True)
).add_to(m)

# --- 3. 自動縮放至數據範圍 ---
m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

# 加入圖層控制
folium.LayerControl().add_to(m)

# 儲存
m.save("preview_drainage.html")
print("✅ 地圖已生成：preview_drainage.html")
print(f"📍 地圖範圍：經度 {bounds[0]:.4f} - {bounds[2]:.4f}, 緯度 {bounds[1]:.4f} - {bounds[3]:.4f}")