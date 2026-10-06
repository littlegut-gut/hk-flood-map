# rainfall_map.py (使用 CSV 站點列表)
import requests
import json
import folium
import geopandas as gpd
import pandas as pd
from datetime import datetime

# --- 1. 從 CSV 讀取站點坐標 ---
def get_station_coords(csv_path="hko_stations.csv"):
    """
    從 CSDI 下載的 CSV 讀取站點資訊
    請確保 CSV 包含以下欄位：stationId, stationName, latitude, longitude
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except FileNotFoundError:
        print(f"❌ 找不到檔案：{csv_path}")
        print("請確保已從 CSDI 下載氣象站 CSV，並命名為 hko_stations.csv")
        return {}
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='big5')  # 有時 CSDI 用 big5
    
    stations = {}
    # 試著自動偵測欄位名稱（常見的變體）
    id_col = None
    name_col = None
    lat_col = None
    lng_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'stationid' in col_lower or 'id' in col_lower:
            id_col = col
        elif 'stationname' in col_lower or 'name' in col_lower:
            name_col = col
        elif 'latitude' in col_lower or 'lat' in col_lower:
            lat_col = col
        elif 'longitude' in col_lower or 'lng' in col_lower or 'lon' in col_lower:
            lng_col = col
    
    if None in (id_col, lat_col, lng_col):
        print("❌ CSV 欄位名稱不符合預期，請檢查檔案欄位。")
        print("目前欄位：", list(df.columns))
        return {}
    
    for _, row in df.iterrows():
        sid = str(row[id_col]).strip()
        name = row[name_col] if name_col else sid
        lat = row[lat_col]
        lng = row[lng_col]
        if sid and pd.notna(lat) and pd.notna(lng):
            stations[sid] = {'name': name, 'lat': float(lat), 'lng': float(lng)}
    
    print(f"✅ 從 CSV 成功讀取 {len(stations)} 個氣象站")
    return stations

# --- 2. 獲取雨量數據（與之前相同）---
def get_rainfall():
    url_rain = "https://data.weather.gov.hk/weatherAPI/opendata/hourlyRainfall.php?lang=zh&station=all"
    try:
        r = requests.get(url_rain, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"❌ 獲取雨量數據失敗：{e}")
        return {}
    rainfall = {}
    for item in data.get('hourlyRainfall', []):
        station_id = item.get('automaticWeatherStationID')
        value = item.get('value')
        if station_id is not None and value is not None:
            try:
                rainfall[station_id] = float(value)
            except ValueError:
                rainfall[station_id] = 0.0
    return rainfall

# --- 3. 主程式 ---
def create_rainfall_map():
    print("📡 從 CSV 讀取站點坐標...")
    stations = get_station_coords()
    if not stations:
        print("⚠️ 沒有站點數據，請確認 hko_stations.csv 檔案存在且格式正確。")
        return
    
    print("🌧️ 獲取雨量數據...")
    rainfall = get_rainfall()
    print(f"✅ 找到 {len(rainfall)} 個雨量記錄")
    
    # 合併
    merged = []
    for sid, info in stations.items():
        rain = rainfall.get(sid, 0.0)
        merged.append({
            'station_id': sid,
            'name': info['name'],
            'lat': info['lat'],
            'lng': info['lng'],
            'rainfall': rain
        })
    print(f"✅ 合併後共 {len(merged)} 個站點有雨量數據")
    
    # --- 載入靜態圖層 (與之前相同) ---
    try:
        pipe = gpd.read_file("data/pipe_rainwater.geojson")
        inlets = gpd.read_file("data/inlets_density.geojson")
    except Exception as e:
        print(f"❌ 讀取靜態數據失敗：{e}")
        return
    
    m = folium.Map(location=[22.3193, 114.1694], zoom_start=11)
    
    folium.GeoJson(
        pipe,
        name="雨水渠管",
        style_function=lambda x: {'color': 'blue', 'weight': 1.5, 'opacity': 0.6}
    ).add_to(m)
    
    def style_function(feature):
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
        tooltip=folium.GeoJsonTooltip(fields=['count'], aliases=['🕳️ 渠蓋數量: '])
    ).add_to(m)
    
    # 雨量站點
    for s in merged:
        rain = s['rainfall']
        radius = max(3, min(20, 5 + rain * 0.5))
        if rain == 0:
            color = 'blue'
            fill_color = 'blue'
        elif rain < 5:
            color = 'green'
            fill_color = 'green'
        elif rain < 15:
            color = 'orange'
            fill_color = 'orange'
        else:
            color = 'red'
            fill_color = 'red'
        
        folium.CircleMarker(
            location=[s['lat'], s['lng']],
            radius=radius,
            popup=f"{s['name']}<br>雨量: {rain} mm",
            color=color,
            fill=True,
            fillColor=fill_color,
            fillOpacity=0.8,
            weight=1
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"rainfall_map_{timestamp}.html"
    m.save(filename)
    print(f"✅ 地圖已儲存：{filename}")

if __name__ == "__main__":
    create_rainfall_map()