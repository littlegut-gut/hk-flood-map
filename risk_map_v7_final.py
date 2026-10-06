# risk_map.py (乾淨重構版)
# 結構：imports → 輔助函數 → 常量 → 數據加載 → TC 解析 → 地圖 → 圖層 → 保存
import os
import json
import folium
import geopandas as gpd
import pandas as pd
import numpy as np
import branca.colormap as cm
import requests
from io import StringIO
from math import radians, sin, cos, sqrt, atan2
from glob import glob

print("🗺️ 開始生成風險地圖...")


# ============================================================
# 常量
# ============================================================
HK_LAT, HK_LON = 22.32, 114.17
ASTRONOMICAL_HIGH_TIDE = 2.0   # 天文高潮典型值（米）
SURGE_BUFFER_M = 3000          # 風暴潮影響圈半徑（米）

TC_INTENSITY_COLORS = {
    'Low Pressure Area':     '#999999', '低壓區':        '#999999',
    'Tropical Depression':   '#4d94ff', '熱帶低氣壓':    '#4d94ff',
    'Tropical Storm':        '#00b050', '熱帶風暴':      '#00b050',
    'Severe Tropical Storm': '#ffcc00', '強烈熱帶風暴':  '#ffcc00',
    'Typhoon':               '#ff6600', '颱風':          '#ff6600',
    'Severe Typhoon':        '#cc0000', '強颱風':        '#cc0000',
    'Super Typhoon':         '#8b0000', '超強颱風':      '#8b0000',
    'Extratropical Low':     '#888888', '溫帶氣旋':      '#888888',
}

# 英文站名 → 中文名（潮汐站）
TIDE_NAME_MAPPING = {
    'Quarry Bay': '鰂魚涌', 'Shek Pik': '石壁', 'Tai O': '大澳',
    'Tsim Bei Tsui': '尖鼻咀', 'Tai Miu Wan': '大廟灣',
    'Tai Po Kau': '大埔滘', 'Waglan Island': '橫瀾島',
    'Cheung Chau': '石壁', 'Chek Lap Kok (E)': '尖鼻咀',
    'Chi Ma Wan': '石壁', 'Kwai Chung': '鰂魚涌',
    'Ko Lau Wan': '大埔滘', 'Lok On Pai': '尖鼻咀',
    'Ma Wan': '大廟灣', 'Po Toi': '橫瀾島',
}


# ============================================================
# 輔助函數
# ============================================================
def sample_points_by_grid(points_gdf, grid_size=200, crs_utm='EPSG:2326'):
    """按 grid_size 網格抽樣，每格保留 1 個點。"""
    if len(points_gdf) == 0:
        return points_gdf
    pts_utm = points_gdf.to_crs(crs_utm)
    x_grid = (pts_utm.geometry.x // grid_size).astype(int)
    y_grid = (pts_utm.geometry.y // grid_size).astype(int)
    pts_utm['_grid_id'] = x_grid.astype(str) + "_" + y_grid.astype(str)
    sampled = pts_utm.drop_duplicates(subset='_grid_id', keep='first').drop(columns='_grid_id')
    print(f"   抽樣：{len(points_gdf)} → {len(sampled)}")
    return sampled.to_crs(epsg=4326)


def _haversine(lat1, lon1, lat2, lon2):
    """兩點大圓距離（km）。"""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _parse_coord_str(s):
    """'24.70N' → 24.70；'145.80E' → 145.80；'22.5S' → -22.5"""
    if pd.isna(s) or s is None:
        return None
    s = str(s).strip()
    try:
        v = float(s[:-1])
        return -v if s[-1] in 'SsWw' else v
    except (ValueError, IndexError):
        return None


def _parse_wind_str(s):
    """'105km/h' → 105.0；'-' → None"""
    if pd.isna(s) or s is None or str(s).strip() in ('-', ''):
        return None
    try:
        return float(''.join(c for c in str(s) if c.isdigit() or c == '.'))
    except ValueError:
        return None


def load_tc_track_from_gml(gml_path):
    """從 OGR GML 讀取 TC 路徑。"""
    try:
        gdf = gpd.read_file(gml_path)
    except Exception as e:
        print(f"  ⚠️ 讀取失敗 {gml_path}: {e}")
        return None
    if len(gdf) == 0:
        return None

    first = gdf.iloc[0]
    name_zh = first.get('TropicalCycloneChineseName', '')
    name_en = first.get('TropicalCycloneEnglishName', '')
    name = name_zh if name_zh else name_en

    # BulletinTime
    bt = ''
    try:
        bt = (f"{int(first['BulletinTime_YEAR'])}-"
              f"{int(first['BulletinTime_MONTH']):02d}-"
              f"{int(first['BulletinTime_DAY']):02d} "
              f"{int(first['BulletinTime_HOUR']):02d}:"
              f"{int(first['BulletinTime_MINUTE']):02d}")
    except (KeyError, ValueError, TypeError):
        pass

    if 'InformationType' not in gdf.columns:
        print(f"  ⚠️ {gml_path} 缺少 InformationType 欄位")
        return None

    pts_df = gdf[gdf['InformationType'].notna()].copy()
    if len(pts_df) == 0:
        return None

    def build_points(sub):
        sub = sub.sort_values('Index') if 'Index' in sub.columns else sub
        out = []
        for _, row in sub.iterrows():
            lat = _parse_coord_str(row.get('Latitude'))
            lon = _parse_coord_str(row.get('Longitude'))
            if lat is None or lon is None:
                try:
                    lat, lon = row.geometry.y, row.geometry.x
                except Exception:
                    continue

            t = ''
            try:
                t = (f"{int(row['Time_YEAR'])}-{int(row['Time_MONTH']):02d}-"
                     f"{int(row['Time_DAY']):02d} "
                     f"{int(row['Time_HOUR']):02d}:{int(row['Time_MINUTE']):02d} "
                     f"{row.get('Time_TIMEZONE', '')}")
            except (KeyError, ValueError, TypeError):
                pass

            out.append({
                'lat':       lat,
                'lon':       lon,
                'intensity': row.get('Intensity', ''),
                'maxwind':   _parse_wind_str(row.get('MaximumWind')),
                'time':      t,
                'index':     row.get('Index'),
            })
        return out

    return {
        'name':          name,
        'bulletin_time': bt,
        'past':          build_points(pts_df[pts_df['InformationType'] == 'PastInformation']),
        'analysis':      build_points(pts_df[pts_df['InformationType'] == 'AnalysisInformation']),
        'forecast':      build_points(pts_df[pts_df['InformationType'] == 'ForecastInformation']),
    }


# ============================================================
# 1. 載入渠管
# ============================================================
print("📂 載入渠管...")
pipe_layers = []
for path, label in [
    ("data/Pipe_Stormwater_SHP/Pipe_Stormwater_SHP.shp", "Pipe_Stormwater_SHP"),
    ("data/Multiple_Pipes_Stormwater/Multiple_Pipes_Stormwater.shp", "Multiple_Pipes_Stormwater"),
]:
    try:
        p = gpd.read_file(path)
        if p.crs is None:
            p = p.set_crs(epsg=2326, allow_override=True)
        p = p.to_crs(epsg=2326)
        p = p[p.geometry.length > 50]
        pipe_layers.append(p)
        print(f"  ✅ {label}：{len(p)} 條（已過濾短管）")
    except Exception as e:
        print(f"  ⚠️ 載入 {label} 失敗：{e}")

if pipe_layers:
    pipe = gpd.GeoDataFrame(
        pd.concat([layer[['geometry']] for layer in pipe_layers], ignore_index=True),
        crs='EPSG:2326'
    )
    pipe['geometry'] = pipe.geometry.simplify(tolerance=2.0, preserve_topology=True)
    pipe = pipe.to_crs(epsg=4326)
    print(f"  ✅ 合併後 {len(pipe)} 條渠管（已簡化）")
else:
    pipe = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')


# ============================================================
# 2. 進水口格柵（抽樣）
# ============================================================
print("📂 載入進水口格柵...")
try:
    gully = gpd.read_file("data/Drain_Gully_Grating_SHP/Drain_Gully_Grating_SHP.shp")
    if gully.crs is None:
        gully = gully.set_crs(epsg=2326, allow_override=True)
    print(f"  原始數量：{len(gully)}")
    gully = sample_points_by_grid(gully, grid_size=400)
    print(f"  ✅ 抽樣後：{len(gully)} 個進水口")
except Exception as e:
    print(f"  ⚠️ 載入進水口格柵失敗：{e}")
    gully = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')


# ============================================================
# 3. 檢查井（抽樣）
# ============================================================
print("📂 載入檢查井...")
try:
    manhole = gpd.read_file("data/Manhole_Stormwater_SHP/Manhole_Stormwater_SHP.shp")
    if manhole.crs is None:
        manhole = manhole.set_crs(epsg=2326, allow_override=True)
    print(f"  原始數量：{len(manhole)}")
    manhole = sample_points_by_grid(manhole, grid_size=400)
    print(f"  ✅ 抽樣後：{len(manhole)} 個檢查井")
except Exception as e:
    print(f"  ⚠️ 載入檢查井失敗：{e}")
    manhole = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')


# ============================================================
# 4. 風險數據
# ============================================================
print("📂 載入風險數據...")
risk = gpd.read_file("data/risk_zones.geojson")
print(f"  ✅ {len(risk)} 個風險網格")
risk_utm = risk.to_crs(epsg=2326)

# ============================================================
# 實時雨量（Python 端初始加載）
# ============================================================
print("🌧️ 載入實時雨量...")

def _fetch_rainfall():
    """從 HKO 取得每小時雨量。"""
    try:
        url = ("https://data.weather.gov.hk/weatherAPI/opendata/"
               "hourlyRainfall.php?lang=zh&station=all")
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ⚠️ 無法獲取雨量：{e}")
        return {}, ""

    rainfall = {}
    update_time = data.get('updateTime', '')
    for item in data.get('hourlyRainfall', []):
        sid = item.get('automaticWeatherStationID')
        val = item.get('value')
        if sid and val is not None:
            try:
                rainfall[sid] = float(val)
            except (ValueError, TypeError):
                rainfall[sid] = 0.0
    return rainfall, update_time


def _load_station_coords(csv_path="hko_stations.csv"):
    """讀取氣象站坐標。"""
    if not os.path.exists(csv_path):
        print(f"  ⚠️ 找不到 {csv_path}")
        return {}
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='big5')
    except Exception as e:
        print(f"  ⚠️ 讀取 CSV 失敗：{e}")
        return {}

    id_col = lat_col = lng_col = name_col = None
    for col in df.columns:
        cl = col.lower()
        if 'stationid' in cl or 'automaticweatherstationid' in cl:
            id_col = col
        elif 'latitude' in cl or 'lat' in cl:
            lat_col = col
        elif 'longitude' in cl or 'lng' in cl or 'lon' in cl:
            lng_col = col
        elif 'stationname' in cl or 'name' in cl:
            name_col = col

    if None in (id_col, lat_col, lng_col):
        print(f"  ❌ CSV 缺少必要欄位。現有：{list(df.columns)}")
        return {}

    stations = {}
    for _, row in df.iterrows():
        sid = str(row[id_col]).strip()
        try:
            lat = float(row[lat_col])
            lng = float(row[lng_col])
        except (ValueError, TypeError):
            continue
        if sid and pd.notna(lat) and pd.notna(lng):
            name = row[name_col] if name_col else sid
            stations[sid] = {'name': name, 'lat': lat, 'lng': lng}
    print(f"  ✅ 氣象站坐標：{len(stations)} 個")
    return stations


rainfall_dict, rainfall_time = _fetch_rainfall()
stations_dict = _load_station_coords()

# 用來前端自動更新的精簡版數據
rain_stations_json = json.dumps([
    {
        'id':   sid,
        'name': info['name'],
        'lat':  info['lat'],
        'lng':  info['lng'],
        'rain': rainfall_dict.get(sid, 0.0),
    }
    for sid, info in stations_dict.items()
], ensure_ascii=False)
rain_stations_json = rain_stations_json.replace('</', '<\\/')

print(f"  ✅ 雨量數據：{len(rainfall_dict)} 站  時間：{rainfall_time or '未知'}")

# ============================================================
# 5. 潮汐站 + 實時潮位
# ============================================================
print("📂 載入潮汐站（含潮位）...")
tide = None
real_tide = {}
tide_time = ""
try:
    tide = gpd.read_file("data/Predicted_tidal_information_Times_and_heights_SHP/HLT_converted.shp")
    if tide.crs is None:
        tide = tide.set_crs(epsg=4326, allow_override=True)
    tide = tide.to_crs(epsg=4326)

    try:
        url = "https://data.weather.gov.hk/weatherAPI/hko_data/tide/ALL_tc.csv"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df_tide = pd.read_csv(StringIO(r.content.decode('utf-8-sig')))
        for _, row in df_tide.iterrows():
            sid = str(row['潮汐站']).strip()
            try:
                real_tide[sid] = float(row['高度(米)'])
            except (ValueError, TypeError):
                pass
        for col in ['日期時間', '日期时间', 'DateTime', 'date_time']:
            if col in df_tide.columns and len(df_tide) > 0:
                tide_time = str(df_tide[col].iloc[0]).strip()
                break
        print(f"  ✅ 實時潮汐：{len(real_tide)} 站  時間: {tide_time or '未知'}")
    except Exception as e:
        print(f"  ⚠️ 無法獲取實時潮汐：{e}")
except Exception as e:
    print(f"  ⚠️ 載入潮汐站失敗：{e}")


# ============================================================
# 6. 熱帶氣旋路徑（必須在繪圖之前定義 tc_tracks）
# ============================================================
print("🌀 載入熱帶氣旋路徑...")
tc_dir = "data/tc"
tc_tracks = []
if os.path.isdir(tc_dir):
    for fp in sorted(glob(os.path.join(tc_dir, "hko_tctrack_*.gml"))
                   + glob(os.path.join(tc_dir, "hko_tctrack_*.xml"))):
        track = load_tc_track_from_gml(fp)
        if track and (track['past'] or track['analysis'] or track['forecast']):
            tc_tracks.append(track)
            print(f"  ✅ {track['name']}（{os.path.basename(fp)}）："
                  f"過去 {len(track['past'])}｜當前 {len(track['analysis'])}｜"
                  f"預報 {len(track['forecast'])}")
        else:
            print(f"  ⚠️ {os.path.basename(fp)}：解析為空")
else:
    print(f"  ℹ️ 找不到 {tc_dir} 目錄")


# ============================================================
# 7. 建立地圖 + 底圖
# ============================================================
m = folium.Map(
    location=[22.3193, 114.1694],
    zoom_start=11,
    tiles=None,
    prefer_canvas=True,    # ← 關鍵：用 Canvas 取代 SVG 渲染
)

# 自訂 popup 樣式
m.get_root().header.add_child(folium.Element("""
<style>
.custom-risk-popup .leaflet-popup-content-wrapper {
    background: rgba(255,255,255,0.98);
    border: 2px solid #444;
    border-radius: 10px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
}
.custom-risk-popup .leaflet-popup-content {
    margin: 12px 14px;
    font-family: 'Segoe UI','Microsoft JhengHei',sans-serif;
}
.custom-risk-popup .leaflet-popup-tip {
    background: rgba(255,255,255,0.98);
    border: 2px solid #444;
}
.risk-popup-body {
    font-family: 'Segoe UI', 'Microsoft JhengHei', sans-serif;
    font-size: 13px;
    line-height: 1.8;
    min-width: 220px;
}
.risk-score {
    color: #c00;
    font-weight: 700;
}
</style>
"""))

# ===== 底圖 1：地形圖（默認，第一個會自動被選中）=====
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Esri Topo',
    name='底圖：地形圖',
    overlay=False,
    control=True,
).add_to(m)

# ===== 底圖 2：衛星影像 =====
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri World Imagery',
    name='底圖：衛星影像',
    overlay=False,
    control=True,
).add_to(m)

# ===== 底圖 3：街道圖（Esri）=====
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Esri Street Map',
    name='底圖：街道圖',
    overlay=False, control=True,
).add_to(m)

# ===== 底圖 4：地形圖（OpenTopoMap）=====
folium.TileLayer(
    tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
    attr='© OpenStreetMap contributors, SRTM | © OpenTopoMap',
    name='底圖：OpenTopoMap',
    overlay=False, control=True,
    subdomains='abc',
    max_zoom=17,
).add_to(m)

# ============================================================
# 8. 繪製各圖層（順序：底 → 上）
# ============================================================
# --- 8.1 渠管（預設關閉）---
pipe_layer = folium.FeatureGroup(name="雨水渠管", show=False)
folium.GeoJson(
    pipe,
    style_function=lambda x: {'color': '#0066cc', 'weight': 1.0, 'opacity': 0.5}
).add_to(pipe_layer)
pipe_layer.add_to(m)

# --- 8.2 進水口格柵 ---
folium.GeoJson(
    gully, name="進水口格柵 (抽樣)",
    marker=folium.CircleMarker(
        radius=1.5, color='#886600', fill=True,
        fillColor='#ffdd00', fillOpacity=0.6, weight=0.3
    )
).add_to(m)

# --- 8.3 檢查井 ---
folium.GeoJson(
    manhole, name="檢查井 (抽樣)",
    marker=folium.CircleMarker(
        radius=1.2, color='#555', fill=True,
        fillColor='#aaa', fillOpacity=0.5, weight=0.3
    )
).add_to(m)

# --- 8.4 風險網格 ---
scores = risk['risk_score']
colormap = cm.LinearColormap(
    ['green', 'yellow', 'red'],
    vmin=scores.min(), vmax=scores.max()
)
colormap.caption = '積水風險分數'


def risk_style(feature):
    s = feature['properties']['risk_score']
    color = '#cccccc' if (s is None or np.isnan(s)) else colormap(s)
    return {
        'fillColor': color,
        'color': 'black',
        'weight': 0.3,
        'fillOpacity': 0.7,
    }

# ============================================================
# 風險網格：格式化欄位 + 因子貢獻 + Popup
# ============================================================
risk_disp = risk.copy()

# --- 基本格式化 ---
risk_disp['_score'] = risk_disp['risk_score'].round(1)
risk_disp['_c']     = (risk_disp['runoff_coeff'].round(2)
                       if 'runoff_coeff' in risk_disp.columns else 0.0)
risk_disp['_elev']  = (risk_disp['elevation'].fillna(0).round(0).astype(int)
                       if 'elevation' in risk_disp.columns
                       else pd.Series([0] * len(risk_disp), index=risk_disp.index))
risk_disp['_pop']   = (risk_disp['population_total'].fillna(0).round(0).astype(int)
                       if 'population_total' in risk_disp.columns
                       else pd.Series([0] * len(risk_disp), index=risk_disp.index))
risk_disp['_inlet'] = (risk_disp['inlet_count'].fillna(0).astype(int)
                       if 'inlet_count' in risk_disp.columns
                       else pd.Series([0] * len(risk_disp), index=risk_disp.index))
risk_disp['_pipe']  = (risk_disp['dist_to_pipe'].fillna(99999).round(0).astype(int)
                       if 'dist_to_pipe' in risk_disp.columns
                       else pd.Series([99999] * len(risk_disp), index=risk_disp.index))
risk_disp['_flood'] = (risk_disp['dist_to_flood'].fillna(99999).round(0).astype(int)
                       if 'dist_to_flood' in risk_disp.columns
                       else pd.Series([99999] * len(risk_disp), index=risk_disp.index))
risk_disp['_coast'] = (risk_disp['coastal_factor'].fillna(0).round(2)
                       if 'coastal_factor' in risk_disp.columns
                       else pd.Series([0.0] * len(risk_disp), index=risk_disp.index))

# --- 因子貢獻（用與 calculate_risk 相同的權重，僅供顯示）---
WEIGHTS_DISPLAY = {
    'inlet':  0.12, 'elev':  0.13, 'pipe':  0.20,
    'pop':    0.25, 'coastal': 0.13, 'runoff': 0.17,
}


def _rank(s):
    return s.rank(pct=True)


# 逐項計算因子貢獻分（0 ~ 權重值）
risk_disp['_f_inlet']   = (1 - _rank(risk_disp['_inlet']))  * WEIGHTS_DISPLAY['inlet']
risk_disp['_f_elev']    = (1 - _rank(risk_disp['_elev']))   * WEIGHTS_DISPLAY['elev']
risk_disp['_f_pipe']    = _rank(risk_disp['_pipe'])         * WEIGHTS_DISPLAY['pipe']
risk_disp['_f_pop']     = _rank(risk_disp['_pop'])          * WEIGHTS_DISPLAY['pop']
risk_disp['_f_coastal'] = risk_disp['_coast'].clip(0, 1)    * WEIGHTS_DISPLAY['coastal']
risk_disp['_f_runoff']  = _rank(risk_disp['_c'])            * WEIGHTS_DISPLAY['runoff']


def _top_factor(row):
    """
    主導因子判定：
    - 水體（C 值 ≥ 0.95）→ 顯示「水體」
    - 郊野（人口 < 1）→ 顯示「海拔」
    - 市區（人口 ≥ 1）→ 找貢獻最大的風險因子
    """
    # 1. 水體
    if row['_c'] >= 0.95:
        return '水體'

    # 2. 郊野（人口 0）：海拔是決定性因子
    if row['_pop'] < 1:
        return '海拔'

    # 3. 市區：找貢獻最大的正向風險因子
    risk_factors = {
        '人口':     row['_f_pop'],
        '徑流 (C)': row['_f_runoff'],
        '渠管':     row['_f_pipe'],
        '渠蓋':     row['_f_inlet'],
        '沿海':     row['_f_coastal'],
    }
    return max(risk_factors, key=risk_factors.get)

risk_disp['_top'] = risk_disp.apply(_top_factor, axis=1)

# ============================================================
# 混合聚合：高風險區（≥50分）保留 200m，低風險區聚合成 400m
# ============================================================
print("  🔧 混合聚合：高風險保留 200m，低風險聚合 400m...")

KEEP_200M_THRESHOLD = 50   # 風險 ≥50 分的網格保留原解析度
AGG_SIZE = 400             # 低風險區聚合到 400m

# --- 高風險：保留 200m 原樣 ---
high = risk_disp[risk_disp['risk_score'] >= KEEP_200M_THRESHOLD].copy()
print(f"     高風險（≥{KEEP_200M_THRESHOLD}分）：{len(high)} 個（保留 200m）")

# --- 低風險：投影到 UTM 後聚合成 400m ---
low = risk_disp[risk_disp['risk_score'] < KEEP_200M_THRESHOLD].copy()
low_utm = low.to_crs(epsg=2326)

# 質心算 400m 網格 ID
low_utm['_gx'] = (low_utm.geometry.centroid.x // AGG_SIZE).astype(int)
low_utm['_gy'] = (low_utm.geometry.centroid.y // AGG_SIZE).astype(int)


def _mode_or_default(series):
    """取眾數，若全空則回傳 '海拔'"""
    if len(series) == 0:
        return '海拔'
    m = series.mode()
    return m.iloc[0] if len(m) > 0 else series.iloc[0]


low_agg = low_utm.dissolve(
    by=['_gx', '_gy'],
    aggfunc={
        'risk_score': 'mean',
        '_score':     'mean',
        '_c':         'mean',
        '_elev':      'mean',
        '_pop':       'sum',      # 人口求和
        '_inlet':     'sum',      # 渠蓋求和
        '_pipe':      'mean',
        '_flood':     'mean',
        '_coast':     'mean',
        '_top':       _mode_or_default,
    }
).reset_index(drop=True).to_crs(epsg=4326)

print(f"     低風險（<{KEEP_200M_THRESHOLD}分）：{len(low)} 個 → 聚合為 {len(low_agg)} 個（400m）")

# --- 合併 ---
risk_out = pd.concat([high, low_agg], ignore_index=True)

# 只保留 popup 需要的欄位
risk_out = risk_out[[
    'geometry',
    'risk_score',
    '_score', '_top', '_c', '_elev', '_pop',
    '_inlet', '_pipe', '_flood', '_coast',
]].copy()

# 壓縮精度
risk_out['risk_score'] = risk_out['risk_score'].round(1)
risk_out['_score']     = risk_out['_score'].round(1)
risk_out['_c']         = risk_out['_c'].round(2)
risk_out['_coast']     = risk_out['_coast'].round(2)

print(f"  ✅ 混合聚合完成：27,209 → {len(risk_out)} 個網格")

m.get_root().header.add_child(folium.Element("""
<style>
.legend.leaflet-control {
    position: fixed !important;
    bottom: 15px !important;
    left: 340px !important;    /* TC 面板右邊 */
    top: auto !important;
    right: auto !important;
    z-index: 900 !important;
}
</style>
"""))

colormap.add_to(m)


# --- 8.6 歷史積水黑點 ---
try:
    flood = gpd.read_file("data/FloodingBlackspots.geojson").to_crs(epsg=4326)
    tt_fields, tt_aliases = [], []
    for f, a in [('LOCATION_TC', '位置: '), ('DISTRICT_TC', '地區: '),
                 ('FLOODING_BLACKSPOT_SCALE', '規模: ')]:
        if f in flood.columns:
            tt_fields.append(f)
            tt_aliases.append(a)

    folium.GeoJson(
        flood, name="歷史積水黑點",
        marker=folium.CircleMarker(
            radius=8, color='#8B0000', fill=True,
            fillColor='#FF0000', fillOpacity=0.9, weight=2
        ),
        tooltip=folium.GeoJsonTooltip(fields=tt_fields, aliases=tt_aliases) if tt_fields else None
    ).add_to(m)
    print(f"  ✅ 歷史黑點：{len(flood)} 個")
except Exception as e:
    print(f"  ⚠️ 載入歷史黑點失敗：{e}")

# --- 8.7 人口密度 ---
try:
    pop = gpd.read_file(
        "data/Land_area_mid_year_population_and_population__SHP/Density_2024_converted.shp"
    ).to_crs(epsg=4326)

    pop_cmap = cm.LinearColormap(
        ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'],
        vmin=pop['POPN_D'].min(), vmax=pop['POPN_D'].max()
    )
    pop_cmap.caption = '人口密度 (人/km²)'

    def pop_style(feature):
        return {
            'fillColor': pop_cmap(feature['properties']['POPN_D']),
            'color': '#666', 'weight': 0.5,
            'fillOpacity': 0.15, 'dashArray': '5, 5'
        }

    folium.GeoJson(
        pop, name="人口密度",
        style_function=pop_style,
        tooltip=folium.GeoJsonTooltip(
            fields=['DC_ENG', 'POPN_D'],
            aliases=['分區: ', '密度(人/km²): ']
        )
    ).add_to(m)
    print(f"  ✅ 人口密度：{len(pop)} 個分區")
except Exception as e:
    print(f"  ⚠️ 載入人口密度失敗：{e}")

# --- 8.8 風暴潮影響圈 ---
print("🌊 計算風暴潮影響圈...")
surge_layer = folium.FeatureGroup(name="風暴潮影響圈 (3km)", show=True)
n_surge = 0
if tide is not None and real_tide:
    for _, row in tide.iterrows():
        en_name = ''
        for col in ['TideStatio', 'TideStation_en', 'TideStation']:
            if col in row.index and pd.notna(row[col]):
                en_name = str(row[col]).strip()
                break

        zh_name = TIDE_NAME_MAPPING.get(en_name, '')
        height = real_tide.get(zh_name)
        if height is None:
            continue

        surge = max(0.0, height - ASTRONOMICAL_HIGH_TIDE)
        if surge <= 0:
            continue

        n_surge += 1
        if surge >= 1.0:
            fill_color, border_color, opacity, level = '#cc0000', '#660000', 0.55, '嚴重'
        elif surge >= 0.5:
            fill_color, border_color, opacity, level = '#ff6600', '#993300', 0.45, '中等'
        else:
            fill_color, border_color, opacity, level = '#ffcc00', '#997700', 0.35, '輕微'

        folium.Circle(
            location=[row.geometry.y, row.geometry.x],
            radius=SURGE_BUFFER_M,
            color=border_color, weight=2,
            fill=True, fillColor=fill_color, fillOpacity=opacity,
            tooltip=folium.Tooltip(
                f"<div style='font-family:sans-serif;font-size:12px;'>"
                f"<b>{zh_name or en_name}</b> 風暴潮影響圈<br>"
                f"當前潮位：<b>{height:.2f} m</b><br>"
                f"增水：<b style='color:{border_color};'>{surge:.2f} m</b>（{level}）<br>"
                f"影響半徑：{SURGE_BUFFER_M/1000:.0f} km</div>",
                sticky=True
            )
        ).add_to(surge_layer)

if n_surge > 0:
    surge_layer.add_to(m)
    print(f"  ✅ 繪製 {n_surge} 個風暴潮影響圈（半徑 3km）")
else:
    print(f"  ℹ️ 當前無風暴潮（所有潮位 ≤ {ASTRONOMICAL_HIGH_TIDE}m）")


# ============================================================
# 9. TC 路徑繪製 + 影響圈 + 與風險疊加
# ============================================================
overlay_stats = []

if tc_tracks:
    tc_layer     = folium.FeatureGroup(name="🌀 熱帶氣旋路徑", show=True)
    tc_impact    = folium.FeatureGroup(name="🌀 TC 影響區 (200km)", show=True)
    tc_highlight = folium.FeatureGroup(name="🔴 路徑上高風險網格", show=False)

    for track in tc_tracks:
        past_pts     = [(p['lat'], p['lon']) for p in track['past']]
        analysis_pts = [(p['lat'], p['lon']) for p in track['analysis']]
        forecast_pts = [(p['lat'], p['lon']) for p in track['forecast']]

        cur = (track['analysis'][0] if track['analysis']
               else track['past'][-1] if track['past']
               else track['forecast'][0])
        main_color = TC_INTENSITY_COLORS.get(cur['intensity'], '#333333')
        cur_dist = _haversine(HK_LAT, HK_LON, cur['lat'], cur['lon'])

        # 實測段
        past_line = past_pts + analysis_pts
        if len(past_line) >= 2:
            folium.PolyLine(past_line, color=main_color, weight=4, opacity=0.9,
                          tooltip=f"🌀 {track['name']}（實測）").add_to(tc_layer)

        # 預報段
        forecast_line = analysis_pts + forecast_pts
        if len(forecast_line) >= 2:
            folium.PolyLine(forecast_line, color=main_color, weight=4, opacity=0.8,
                          dash_array='12,8',
                          tooltip=f"🌀 {track['name']}（預測）").add_to(tc_layer)

        # 過去點
        for p in track['past']:
            folium.CircleMarker(
                [p['lat'], p['lon']], radius=3, color='#444', weight=1,
                fill=True, fillColor='#cccccc', fillOpacity=0.85,
                tooltip=(f"<b>{track['name']}</b>（過去）<br>"
                         f"時間：{p['time'] or '--'}<br>"
                         f"強度：{p['intensity'] or '--'}<br>"
                         f"風速：{int(p['maxwind']) if p['maxwind'] else '--'} km/h")
            ).add_to(tc_layer)

        # 當前位置
        for p in track['analysis']:
            folium.CircleMarker(
                [p['lat'], p['lon']], radius=10, color='#000', weight=2.5,
                fill=True, fillColor=main_color, fillOpacity=1.0,
                tooltip=(f"<b>🌀 {track['name']}</b>（當前）<br>"
                         f"時間：{p['time'] or '--'}<br>"
                         f"強度：{p['intensity'] or '--'}<br>"
                         f"風速：{int(p['maxwind']) if p['maxwind'] else '--'} km/h<br>"
                         f"距香港：<b>{cur_dist:.0f} km</b>")
            ).add_to(tc_layer)
            if p['maxwind']:
                folium.Marker(
                    [p['lat'], p['lon']],
                    icon=folium.DivIcon(
                        html=(f'<div style="font-size:11px;font-weight:700;'
                              f'color:#fff;background:{main_color};'
                              f'padding:2px 6px;border-radius:3px;'
                              f'white-space:nowrap;'
                              f'box-shadow:0 1px 3px rgba(0,0,0,0.5);">'
                              f'{int(p["maxwind"])} km/h</div>'),
                        icon_size=(80, 18), icon_anchor=(-12, 9),
                    )
                ).add_to(tc_layer)

        # 預報點
        for p in track['forecast']:
            ic = TC_INTENSITY_COLORS.get(p['intensity'], main_color)
            folium.CircleMarker(
                [p['lat'], p['lon']], radius=5, color=ic, weight=2,
                fill=True, fillColor='#fff', fillOpacity=0.9,
                tooltip=(f"<b>{track['name']}</b>（預測）<br>"
                         f"時間：{p['time'] or '--'}<br>"
                         f"強度：{p['intensity'] or '--'}<br>"
                         f"風速：{int(p['maxwind']) if p['maxwind'] else '--'} km/h")
            ).add_to(tc_layer)

        # 200km 影響圈
        for p in track['forecast']:
            folium.Circle(
                [p['lat'], p['lon']], radius=200000,
                color=main_color, weight=1, opacity=0.4,
                fill=True, fillColor=main_color, fillOpacity=0.06,
                tooltip=f"{track['name']} 預測影響區（200km）"
            ).add_to(tc_impact)

        # 與風險疊加
        try:
            from shapely.geometry import LineString
            line_pts = past_pts + analysis_pts + forecast_pts
            if len(line_pts) >= 2:
                line_wgs84 = LineString([(lon, lat) for lat, lon in line_pts])
                line_gdf = gpd.GeoDataFrame(
                    [{'geometry': line_wgs84}], crs='EPSG:4326'
                ).to_crs(epsg=2326)
                buf_100km = line_gdf.geometry.iloc[0].buffer(100000)

                mask = risk_utm.geometry.intersects(buf_100km)
                high_risk = risk_utm[mask & (risk_utm['risk_score'] > 60)]

                if len(high_risk) > 0:
                    high_risk_wgs = high_risk.to_crs(epsg=4326)
                    folium.GeoJson(
                        high_risk_wgs[['risk_score', 'geometry']],
                        style_function=lambda x: {
                            'color': '#ff0000', 'weight': 2.5,
                            'fillColor': '#ff0000', 'fillOpacity': 0.35,
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=['risk_score'],
                            aliases=['路徑區高風險: ']
                        )
                    ).add_to(tc_highlight)

                overlay_stats.append({
                    'name':      track['name'],
                    'distance':  cur_dist,
                    'intensity': cur['intensity'] or '--',
                    'maxwind':   int(cur['maxwind']) if cur['maxwind'] else None,
                    'n_high':    len(high_risk),
                })
        except Exception as e:
            print(f"  ⚠️ 路徑疊加失敗：{e}")

    tc_layer.add_to(m)
    tc_impact.add_to(m)
    if any(s['n_high'] > 0 for s in overlay_stats):
        tc_highlight.add_to(m)
    print(f"  ✅ 繪製 {len(tc_tracks)} 條路徑 + 影響區")


# ============================================================
# 10. 潮位面板（右上角）
# ============================================================
if tide is not None and real_tide:
    panel_rows = []
    for _, row in tide.iterrows():
        en_name = ''
        for col in ['TideStatio', 'TideStation_en', 'TideStation']:
            if col in row.index and pd.notna(row[col]):
                en_name = str(row[col]).strip()
                break
        zh_name = TIDE_NAME_MAPPING.get(en_name, '')
        height = real_tide.get(zh_name)
        if height is None:
            continue
        surge = max(0.0, height - ASTRONOMICAL_HIGH_TIDE)
        panel_rows.append((zh_name or en_name, height, surge))

    if panel_rows:
        panel_rows.sort(key=lambda x: -x[1])
        max_surge = max((s for _, _, s in panel_rows), default=0.0)

        if max_surge >= 0.5:
            header_bg, header_txt = '#cc0000', f'🌊 實時潮位 ⚠️ 增水 {max_surge:.2f}m'
        elif max_surge > 0:
            header_bg, header_txt = '#ff6600', f'🌊 實時潮位 ⚠️ 增水 {max_surge:.2f}m'
        else:
            header_bg, header_txt = '#003366', '🌊 實時潮位'

        rows_html = ""
        for name, h, s in panel_rows:
            if s >= 1.0:
                c, tag = '#cc0000', '嚴重'
            elif s >= 0.5:
                c, tag = '#ff6600', '中等'
            elif s > 0:
                c, tag = '#ddaa00', '輕微'
            else:
                c, tag = '#0099ff', ''
            surge_txt = (f"<span style='color:{c};'>+{s:.2f}m <small>{tag}</small></span>"
                         if s > 0 else "")
            rows_html += (
                f"<tr>"
                f"<td style='padding:2px 8px;'>{name}</td>"
                f"<td style='padding:2px 8px;text-align:right;color:{c};font-weight:600;'>{h:.2f}m</td>"
                f"<td style='padding:2px 8px;text-align:right;font-size:11px;'>{surge_txt}</td>"
                f"</tr>"
            )

        time_str = tide_time if tide_time else '--'
        panel_html = (
            '<div id="tide-panel" style="'
            'position:absolute;top:12px;right:12px;z-index:9999;'
            'background:rgba(255,255,255,0.96);border:1px solid #888;'
            'border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.28);'
            "font-family:'Segoe UI','Microsoft JhengHei',sans-serif;"
            'font-size:12px;min-width:240px;max-width:300px;'
            'overflow:hidden;backdrop-filter:blur(4px);">'
            '<div style="background:' + header_bg + ';color:white;padding:7px 12px;'
            'font-weight:600;font-size:13px;cursor:pointer;user-select:none;"'
            ' onclick="var b=document.getElementById(\'tide-panel-body\');'
            'if(b.style.display===\'none\')b.style.display=\'block\';'
            'else b.style.display=\'none\';"'
            '>' + header_txt + '</div>'
            '<div id="tide-panel-body">'
            '<div style="padding:3px 10px 2px;color:#666;font-size:10.5px;'
            'text-align:right;border-bottom:1px solid #eee;">' + time_str + '</div>'
            '<table style="width:100%;border-collapse:collapse;">'
            '<thead><tr style="background:#eef3f8;color:#333;">'
            '<th style="padding:3px 8px;text-align:left;font-size:11px;">站點</th>'
            '<th style="padding:3px 8px;text-align:right;font-size:11px;">潮位</th>'
            '<th style="padding:3px 8px;text-align:right;font-size:11px;">增水</th>'
            '</tr></thead><tbody>' + rows_html + '</tbody></table></div></div>'
        )
        m.get_root().html.add_child(folium.Element(panel_html))


# ============================================================
# 11. TC 情報面板（左下角）+ 800km 警戒圈 + 首次聲明
# ============================================================
if tc_tracks:
    tc_infos = []
    for track in tc_tracks:
        cur = (track['analysis'][0] if track['analysis']
               else track['past'][-1] if track['past']
               else track['forecast'][0])
        dist = _haversine(HK_LAT, HK_LON, cur['lat'], cur['lon'])
        tc_infos.append({
            'name':          track['name'],
            'intensity':     cur['intensity'] or '--',
            'maxwind':       cur['maxwind'],
            'distance':      dist,
            'bulletin_time': track['bulletin_time'] or '--',
        })

    tc_infos.sort(key=lambda x: x['distance'])
    min_dist = tc_infos[0]['distance']

    if min_dist < 400:
        header_bg, header_txt = '#cc0000', f'🌀 熱帶氣旋情報 ⚠️ 最近 {min_dist:.0f} km'
    elif min_dist < 800:
        header_bg, header_txt = '#ff6600', f'🌀 熱帶氣旋情報 ⚠️ 最近 {min_dist:.0f} km'
    else:
        header_bg, header_txt = '#4a148c', '🌀 熱帶氣旋情報'

    rows_html = ""
    for info in tc_infos:
        color = TC_INTENSITY_COLORS.get(info['intensity'], '#333')
        wind_txt = (str(int(info['maxwind'])) + ' km/h') if info['maxwind'] else '--'
        dist = info['distance']
        dist_color = ('#cc0000' if dist < 400
                      else '#ff6600' if dist < 800 else '#666')
        row_bg = 'background:#fff3e0;' if dist < 800 else ''
        warn_tag = ('<span style="color:#cc0000;font-weight:700;font-size:10px;">'
                    '● 800km內</span>') if dist < 800 else ''
        bt = info['bulletin_time']
        bt_short = bt[11:16] if len(bt) >= 16 else bt

        rows_html += (
            "<div style='padding:7px 10px;border-bottom:1px solid #eee;" + row_bg + "'>"
            "<div style='display:flex;justify-content:space-between;align-items:center;'>"
            "<span style='font-weight:700;color:" + color + ";'>🌀 " + info['name'] + "</span>"
            "<span style='color:" + dist_color + ";font-size:11px;font-weight:600;'>"
            + str(int(dist)) + " km</span>"
            "</div>"
            "<div style='font-size:11px;color:#444;margin-top:2px;'>"
            + info['intensity'] + "　·　" + wind_txt + "　" + warn_tag
            + "</div>"
            "<div style='font-size:10px;color:#999;margin-top:1px;'>更新 " + bt_short + "</div>"
            "</div>"
        )

    tc_infos_json = json.dumps([{
        'name':      t['name'],
        'distance':  round(t['distance'], 1),
        'intensity': t['intensity'],
        'maxwind':   int(t['maxwind']) if t['maxwind'] else None,
    } for t in tc_infos], ensure_ascii=False)
    tc_infos_json = tc_infos_json.replace('</', '<\\/')

    js_template = r"""
(function() {
    var TC_INFOS = __TC_INFOS__;
    var STORAGE_KEY = 'hko_tc_notified_v1';
    var DAY_MS = 24 * 60 * 60 * 1000;
    var now = Date.now();

    var notified = {};
    try { notified = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); }
    catch (e) { notified = {}; }
    for (var k in notified) { if (now - notified[k] > 7 * DAY_MS) delete notified[k]; }

    function bracketOf(d) {
        if (d < 400) return 'danger';
        if (d < 800) return 'warn';
        return 'far';
    }

    var fresh = [];
    TC_INFOS.forEach(function(tc) {
        var bracket = bracketOf(tc.distance);
        if (bracket === 'far') return;
        var key = tc.name + '|' + bracket;
        if (!notified[key]) { fresh.push(tc); notified[key] = now; }
    });
    localStorage.setItem(STORAGE_KEY, JSON.stringify(notified));
    fresh.forEach(function(tc, i) { setTimeout(function() { showTCToast(tc); }, i * 500); });

    function showTCToast(tc) {
        var isDanger = tc.distance < 400;
        var bg = isDanger ? 'linear-gradient(135deg,#cc0000 0%,#6b0000 100%)'
                          : 'linear-gradient(135deg,#ff6600 0%,#cc3300 100%)';
        var title = isDanger ? '⚠️ 熱帶氣旋嚴重警戒' : '⚠️ 熱帶氣旋警戒';
        var body  = isDanger ? '已進入香港 <b>400 km</b> 範圍'
                             : '已進入香港 <b>800 km</b> 範圍';
        var toast = document.createElement('div');
        toast.style.cssText = [
            'position:fixed','top:20px','left:50%',
            'transform:translateX(-50%) translateY(-20px)',
            'background:' + bg, 'color:#fff',
            'padding:14px 22px','border-radius:10px',
            "font-family:'Segoe UI','Microsoft JhengHei',sans-serif",
            'box-shadow:0 6px 24px rgba(0,0,0,0.45)',
            'z-index:99999','max-width:90vw','text-align:center',
            'opacity:0','transition:opacity .35s,transform .35s',
            'cursor:pointer'
        ].join(';');
        var windTxt = tc.maxwind ? ('　·　' + tc.maxwind + ' km/h') : '';
        toast.innerHTML =
            '<div style="font-size:20px;margin-bottom:4px;">' + title + '</div>' +
            '<div style="font-size:16px;font-weight:700;">🌀 ' + tc.name + '</div>' +
            '<div style="font-size:13px;margin-top:6px;opacity:0.95;">' +
              body + '　·　當前 <b>' + tc.distance.toFixed(0) + ' km</b>' + windTxt +
            '</div>' +
            '<div style="font-size:11px;margin-top:4px;opacity:0.75;">' +
              tc.intensity + '　·　點擊關閉</div>';
        function dismiss() {
            if (!toast.parentNode) return;
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(-50%) translateY(-20px)';
            setTimeout(function() { toast.remove(); }, 350);
        }
        toast.onclick = dismiss;
        document.body.appendChild(toast);
        requestAnimationFrame(function() {
            toast.style.opacity = '1';
            toast.style.transform = 'translateX(-50%) translateY(0)';
        });
        setTimeout(dismiss, isDanger ? 20000 : 12000);
    }
})();
"""
    js_code = js_template.replace('__TC_INFOS__', tc_infos_json)

    tc_panel = (
        '<div id="tc-panel" style="'
        'position:absolute;bottom:30px;left:12px;z-index:9999;'
        'background:rgba(255,255,255,0.96);border:1px solid #888;'
        'border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.28);'
        "font-family:'Segoe UI','Microsoft JhengHei',sans-serif;"
        'font-size:12px;min-width:230px;max-width:290px;'
        'overflow:hidden;backdrop-filter:blur(4px);">'
        '<div style="background:' + header_bg + ';color:white;padding:7px 12px;'
        'font-weight:600;font-size:13px;cursor:pointer;user-select:none;"'
        ' onclick="var b=document.getElementById(\'tc-panel-body\');'
        'var a=document.getElementById(\'tc-panel-arrow\');'
        'if(b.style.display===\'none\'){b.style.display=\'block\';a.textContent=\'▼\';}'
        'else{b.style.display=\'none\';a.textContent=\'▲\';}">'
        '<span>' + header_txt + '</span>'
        '<span id="tc-panel-arrow" style="float:right;font-size:10px;opacity:0.85;">▼</span>'
        '</div>'
        '<div id="tc-panel-body">' + rows_html + '</div>'
        '</div>'
        '<script>' + js_code + '</script>'
    )
    m.get_root().html.add_child(folium.Element(tc_panel))
    print(f"  ✅ TC 面板：{len(tc_tracks)} 條風暴，最近 {min_dist:.0f} km")

    if min_dist < 800:
        folium.Circle(
            location=[HK_LAT, HK_LON], radius=800000,
            color='#ff6600', weight=2, opacity=0.55,
            dash_array='10, 8',
            fill=True, fillColor='#ff6600', fillOpacity=0.04,
            tooltip='800 km 警戒線（以香港為中心）'
        ).add_to(m)
        print(f"     ✅ 繪製 800km 警戒圈")
        print(f"     ⚠️ 有風暴進入 800km 警戒範圍")


# ============================================================
# 12. TC × 風險疊加面板（右下角）
# ============================================================
if tc_tracks and overlay_stats:
    panel_rows = ""
    for s in overlay_stats:
        dist = s['distance']
        dist_color = ('#cc0000' if dist < 400
                      else '#ff6600' if dist < 800 else '#666')
        color = TC_INTENSITY_COLORS.get(s['intensity'], '#333')
        wind_txt = f"{s['maxwind']} km/h" if s['maxwind'] else "--"

        if s['n_high'] > 0:
            warn_tag = (f"<div style='font-size:10px;color:#cc0000;"
                        f"font-weight:700;margin-top:3px;'>"
                        f"🔴 路徑 100km 內高風險網格：{s['n_high']} 個</div>")
        else:
            warn_tag = (f"<div style='font-size:10px;color:#666;"
                        f"margin-top:3px;'>路徑 100km 內無高風險網格</div>")

        panel_rows += (
            f"<div style='padding:7px 10px;border-bottom:1px solid #eee;'>"
            f"<div style='display:flex;justify-content:space-between;'>"
            f"  <span style='font-weight:700;color:{color};'>🌀 {s['name']}</span>"
            f"  <span style='color:{dist_color};font-size:11px;font-weight:600;'>"
            f"    {dist:.0f} km</span>"
            f"</div>"
            f"<div style='font-size:11px;color:#444;margin-top:2px;'>"
            f"  {s['intensity']}　·　{wind_txt}</div>"
            f"{warn_tag}</div>"
        )

    panel = (
        '<div style="position:absolute;bottom:30px;right:12px;z-index:9999;'
        'background:rgba(255,255,255,0.96);border:1px solid #888;'
        'border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.28);'
        "font-family:'Segoe UI','Microsoft JhengHei',sans-serif;"
        'font-size:12px;min-width:240px;max-width:300px;overflow:hidden;'
        'backdrop-filter:blur(4px);">'
        '<div style="background:#4a148c;color:white;padding:7px 12px;'
        'font-weight:600;font-size:13px;">🌀 TC × 風險疊加</div>'
        + panel_rows + '</div>'
    )
    m.get_root().html.add_child(folium.Element(panel))
    print(f"  ✅ TC 疊加面板已生成")


# ============================================================
# 實時雨量圖層 + 面板（全部由 JS 生成）
# ============================================================
if stations_dict:
    map_var_name = m.get_name()   # folium 生成的 map 變數名（例如 map_abc123）

    rain_html = (
        # ---------- 面板 HTML ----------
        '<div id="rain-panel" style="'
        'position:absolute;top:12px;left:50%;transform:translateX(-50%);'
        'z-index:9999;background:rgba(255,255,255,0.96);'
        'border:1px solid #888;border-radius:8px;'
        'box-shadow:0 2px 10px rgba(0,0,0,0.28);'
        "font-family:'Segoe UI','Microsoft JhengHei',sans-serif;"
        'font-size:12px;min-width:260px;max-width:320px;overflow:hidden;'
        'backdrop-filter:blur(4px);">'
        '<div style="background:#0066cc;color:white;padding:7px 12px;'
        'font-weight:600;font-size:13px;cursor:pointer;user-select:none;'
        'display:flex;justify-content:space-between;align-items:center;"'
        ' onclick="var b=document.getElementById(\'rain-body\');'
        'var a=document.getElementById(\'rain-arrow\');'
        'if(b.style.display===\'none\'){b.style.display=\'block\';a.textContent=\'▼\';}'
        'else{b.style.display=\'none\';a.textContent=\'▲\';}">'
        '<span>🌧️ 實時雨量 <span id="rain-status" '
        'style="font-size:10px;opacity:0.85;font-weight:400;">初始</span></span>'
        '<span id="rain-arrow" style="font-size:10px;opacity:0.85;">▼</span>'
        '</div>'
        '<div id="rain-body">'
        '<div id="rain-time" style="padding:3px 10px 2px;color:#666;'
        'font-size:10.5px;text-align:right;border-bottom:1px solid #eee;">'
        + (rainfall_time or '--') + '</div>'
        '<table style="width:100%;border-collapse:collapse;">'
        '<thead><tr style="background:#eef3f8;color:#333;">'
        '<th style="padding:3px 8px;text-align:left;font-size:11px;">氣象站</th>'
        '<th style="padding:3px 8px;text-align:right;font-size:11px;">雨量(mm)</th>'
        '</tr></thead>'
        '<tbody id="rain-tbody"></tbody></table>'
        '<div style="padding:4px 10px;color:#999;font-size:10px;'
        'border-top:1px solid #eee;text-align:center;">'
        '每 10 分鐘自動更新 · '
        '<a href="#" id="rain-refresh" '
        'style="color:#0066cc;text-decoration:none;">立即刷新</a>'
        '</div></div></div>'

        # ---------- JS ----------
        '<script>'
        '(function(){'
        'var STATIONS = ' + rain_stations_json + ';'
        'var API = "https://data.weather.gov.hk/weatherAPI/opendata/hourlyRainfall.php?lang=zh&station=all";'
        'var REFRESH_MS = 10*60*1000;'
        'var mapObj = null;'
        'var _tries = 0;'

	'function _findMap(){'
	'  var candidates = [];'
	'  var exact = window["' + map_var_name + '"];'
	'  if (exact) candidates.push(exact);'
	'  for (var k in window) {'
	'    if (k.indexOf("map_") === 0) {'
	'      try { candidates.push(window[k]); } catch(e) {}'
	'    }'
	'  }'
	'  if (typeof L !== "undefined" && L.Map && L.Map._instances) {'
	'    for (var id in L.Map._instances) {'
	'      try { candidates.push(L.Map._instances[id]); } catch(e) {}'
	'    }'
	'  }'
	'  for (var i = 0; i < candidates.length; i++) {'
	'    var c = candidates[i];'
	'    if (c && typeof c.addLayer === "function" && c._container) return c;'
	'  }'
	'  return null;'
	'}'

	'function _boot(){'
	'  mapObj = _findMap();'
	'  if (mapObj) {'
	'    console.log("[rain] map 就緒，開始初始化");'
	'    _init(mapObj);'
	'  } else if (_tries++ < 50) {'
	'    setTimeout(_boot, 100);'
	'  } else {'
	'    console.error("[rain] 5 秒內找不到 map");'
	'  }'
	'}'
	'function _init(mapObj){'

        # 找到雨量圖層群組（Leaflet LayerControl 需要）
        'var rainLayer = L.layerGroup().addTo(mapObj);'
        'var markers = {};'   # sid -> L.circleMarker

        # 顏色 & 半徑
        'function rainColor(v){'
        '  if(v<=0)return"#cccccc";if(v<2)return"#a8d8ff";if(v<10)return"#4d94ff";'
        '  if(v<30)return"#00b050";if(v<50)return"#ffcc00";'
        '  if(v<80)return"#ff6600";return"#cc0000";}'
        'function radiusOf(v){return 5+Math.min(v/5,10);}'

        # 創建所有 marker
        'STATIONS.forEach(function(s){'
        '  var m = L.circleMarker([s.lat, s.lng], {'
        '    radius: radiusOf(s.rain),'
        '    color: "#333", weight: 1,'
        '    fillColor: rainColor(s.rain),'
        '    fillOpacity: 0.85'
        '  });'
        '  m.bindTooltip("<b>"+s.name+"</b><br>雨量：<b>"+s.rain.toFixed(1)+" mm</b>",'
        '    {sticky:true});'
        '  m.addTo(rainLayer);'
        '  markers[s.id] = m;'
        '});'

        # 更新面板
        'function updatePanel(rain, time){'
        '  document.getElementById("rain-time").textContent = time || "--";'
        '  var arr = Object.keys(rain).map(function(k){'
        '    return {sid:k, v:rain[k]};'
        '  }).sort(function(a,b){return b.v-a.v;}).slice(0,8);'
        '  var html = "";'
        '  arr.forEach(function(e){'
        '    var st = null;'
        '    for(var i=0;i<STATIONS.length;i++){'
        '      if(STATIONS[i].id===e.sid){st=STATIONS[i];break;}'
        '    }'
        '    var name = st ? st.name : e.sid;'
        '    var c = rainColor(e.v);'
        '    html += "<tr><td style=\'padding:2px 8px;\'>"+name+"</td>"'
        '         +  "<td style=\'padding:2px 8px;text-align:right;color:"+c'
        '         +  ";font-weight:600;\'>"+e.v.toFixed(1)+"</td></tr>";'
        '  });'
        '  document.getElementById("rain-tbody").innerHTML = html;'
        '}'

        # 更新 marker
        'function updateMarkers(rain){'
        '  STATIONS.forEach(function(s){'
        '    var v = (rain[s.id]!==undefined) ? rain[s.id] : s.rain;'
        '    var mk = markers[s.id];'
        '    if (!mk) return;'
        '    mk.setStyle({fillColor: rainColor(v)});'
        '    mk.setRadius(radiusOf(v));'
        '    mk.setTooltipContent("<b>"+s.name+"</b><br>雨量：<b>"+v.toFixed(1)+" mm</b>");'
        '  });'
        '}'

        # 初始化面板
        'var initRain = {};STATIONS.forEach(function(s){initRain[s.id]=s.rain;});'
        'updatePanel(initRain, "' + (rainfall_time or '--') + '");'

        # 狀態 & toast
        'function setStatus(t){'
        '  var e=document.getElementById("rain-status");if(e)e.textContent=t;}'
        'function toast(msg,ok){'
        '  var t=document.createElement("div");'
        '  t.style.cssText="position:fixed;bottom:20px;left:50%;transform:translateX(-50%);"'
        '   +"background:"+(ok?"#00b050":"#cc0000")+";color:#fff;padding:8px 16px;"'
        '   +"border-radius:6px;z-index:99999;font-family:sans-serif;font-size:12px;"'
        '   +"box-shadow:0 4px 12px rgba(0,0,0,0.3);opacity:0;transition:opacity 0.3s;";'
        '  t.textContent=msg;document.body.appendChild(t);'
        '  requestAnimationFrame(function(){t.style.opacity="1";});'
        '  setTimeout(function(){t.style.opacity="0";'
        '    setTimeout(function(){t.remove();},300);},2500);}'

        # 抓取
        'function fetchRain(showTip){'
        '  setStatus("更新中…");'
        '  fetch(API).then(function(r){if(!r.ok)throw new Error("HTTP "+r.status);return r.json();})'
        '  .then(function(data){'
        '    var rain={},t=data.obsTime||"";'
        '    (data.hourlyRainfall||[]).forEach(function(it){'
        '      var sid=it.automaticWeatherStationID,v=it.value;'
        '      if(sid&&v!==undefined&&v!==null){'
        '        var f=parseFloat(v);rain[sid]=isNaN(f)?0:f;}'
        '    });'
        '    updatePanel(rain,t);updateMarkers(rain);'
        '    setStatus("已更新");'
        '    if(showTip)toast("✓ 雨量已更新",true);'
        '  })'
        '  .catch(function(e){'
        '    console.error("雨量失敗:",e);setStatus("失敗");'
        '    if(showTip)toast("⚠️ "+e.message,false);'
        '  });'
        '}'

        # 綁定
        'document.getElementById("rain-refresh").onclick=function(e){'
        '  e.preventDefault();fetchRain(true);};'
        'setInterval(function(){fetchRain(false);},REFRESH_MS);'
        'setTimeout(function(){fetchRain(false);},30000);'

        '}'   # _init 的結尾
	'_boot();'
	'})();'
        '</script>'
    )
    m.get_root().html.add_child(folium.Element(rain_html))
    print(f"  ✅ 雨量面板 + JS markers（{len(stations_dict)} 站）")

# ============================================================
# JS：為風險網格綁定 popup + tooltip（動態生成 HTML）
# ============================================================
m.get_root().html.add_child(folium.Element("""
<script>
(function(){
    function makePopupHTML(p) {
        return '<div class="risk-popup-body">'
             + '<b>🎯 風險分數</b>：<span class="risk-score">'
             + p._score + '</span><br>'
             + '<b>📌 主導因子</b>：' + p._top + '<br>'
             + '<b>🏙️ 徑流係數 (C)</b>：' + p._c + '<br>'
             + '<b>⛰️ 海拔 (m)</b>：' + p._elev + '<br>'
             + '<b>👥 人口</b>：' + p._pop + '<br>'
             + '<b>🕳️ 渠蓋數</b>：' + p._inlet + '<br>'
             + '<b>🔧 距渠管 (m)</b>：' + p._pipe + '<br>'
             + '<b>💧 距黑點 (m)</b>：' + p._flood + '<br>'
             + '<b>🏖️ 沿海脆弱性</b>：' + p._coast
             + '</div>';
    }

    function bindRiskPopups(){
        var count = 0;

        function walk(layer) {
            // 檢查是否有 _score 屬性（風險網格的特徵）
            if (layer.feature &&
                layer.feature.properties &&
                layer.feature.properties._score !== undefined &&
                typeof layer.bindPopup === 'function') {
                // 只在第一次訪問時綁定
                if (!layer.getPopup()) {
                    layer.bindPopup(
                        makePopupHTML(layer.feature.properties),
                        { maxWidth: 340, className: 'custom-risk-popup' }
                    );
                    layer.bindTooltip(
                        '風險分數: ' + (layer.feature.properties._score || '--'),
                        { sticky: true }
                    );
                    count++;
                }
            }
            // 遞歸子圖層
            if (layer.eachLayer) {
                layer.eachLayer(walk);
            }
        }

        // 找到 Leaflet map 物件
        var mapObj = null;
        for (var k in window) {
            if (k.indexOf('map_') === 0) {
                var candidate = window[k];
                if (candidate && typeof candidate.eachLayer === 'function'
                    && typeof candidate.addLayer === 'function') {
                    mapObj = candidate;
                    break;
                }
            }
        }

        if (!mapObj) {
            console.error('[risk-popup] 找不到 Leaflet map');
            return;
        }

        mapObj.eachLayer(walk);
        console.log('[risk-popup] 已綁定 ' + count + ' 個風險網格');
    }

    setTimeout(bindRiskPopups, 800);
})();
</script>
"""))

# ============================================================
# 風險圖層最後加入（確保在最上層，可接收點擊）
# ============================================================
folium.GeoJson(
    risk_out,
    name="積水風險",
    style_function=risk_style,
    smooth_factor=2.0,
    prefer_canvas=True,    # ← 明確指定
).add_to(m)

# ============================================================
# 13. 圖層控制 + 保存
# ============================================================
folium.LayerControl(collapsed=False, position='topleft').add_to(m)

m.save("risk_map.html")

size_mb = os.path.getsize("risk_map.html") / 1024 / 1024
print(f"\n✅ 風險地圖已儲存：risk_map.html ({size_mb:.2f} MB)")