# risk_calculator.py (完整版，自動處理所有步驟)
import os
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point
import requests
from rasterio.transform import rowcol
import json

# ===================== 設定 =====================
# 請修改為你實際的 DEM 檔案路徑
DEM_PATH = r"D:\hk_flood_map\data\DEM\Digital Terrain Model.tif"

# ===================== 1. 讀取 DEM =====================
def load_dem(dem_path):
    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"找不到 DEM 檔案：{dem_path}")
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(float)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
    return dem_data, transform, bounds, crs

# ===================== 2. 建立 500m 網格 (HK1980) =====================
def create_grid(bounds, cell_size=500):
    xmin, ymin, xmax, ymax = bounds
    xs = np.arange(xmin, xmax, cell_size)
    ys = np.arange(ymin, ymax, cell_size)
    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]
    grid = gpd.GeoDataFrame({'geometry': cells}, crs='EPSG:2326')
    return grid

# ===================== 3. 提取海拔 =====================
def extract_elevation(grid, dem_data, transform):
    elevations = []
    for idx, row in grid.iterrows():
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        try:
            r, c = rowcol(transform, cx, cy)
            if 0 <= r < dem_data.shape[0] and 0 <= c < dem_data.shape[1]:
                elev = dem_data[r, c]
            else:
                elev = np.nan
        except:
            elev = np.nan
        elevations.append(elev)
    grid['elevation'] = elevations
    grid = grid.dropna(subset=['elevation'])
    
    # ======= 关键修改：剔除海洋/潮間帶 =======
    grid = grid[grid['elevation'] > 2.0]   # 2米以下视为非陆地
    grid['elevation'] = grid['elevation'].astype(float)
    return grid

# ===================== 4. 獲取雨量數據 =====================
def get_rainfall_data():
    url = "https://data.weather.gov.hk/weatherAPI/opendata/hourlyRainfall.php?lang=zh&station=all"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except:
        print("⚠️ 無法獲取雨量數據，將使用模擬數據 (全部為 0)")
        return {}
    rainfall = {}
    for item in data.get('hourlyRainfall', []):
        sid = item.get('automaticWeatherStationID')
        val = item.get('value')
        if sid and val is not None:
            try:
                rainfall[sid] = float(val)
            except:
                rainfall[sid] = 0.0
    return rainfall

# ===================== 5. 讀取站點坐標 (從 CSV) =====================
def load_station_coords(csv_path="hko_stations.csv"):
    """
    从 HKO CSDI 格式的 CSV 读取站点坐标
    支持字段：automaticWeatherStationID, GeometryLongitude, GeometryLatitude
    """
    if not os.path.exists(csv_path):
        print(f"⚠️ 找不到 {csv_path}，将使用空白站点")
        return {}
    
    # 尝试 utf-8，若失败则尝试 big5（您的示例是英文+中文，很可能是 utf-8）
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='big5')
    except Exception as e:
        print(f"⚠️ 读取 CSV 失败：{e}")
        return {}
    
    stations = {}
    # 找出列名（不区分大小写）
    id_col = None
    lat_col = None
    lng_col = None
    name_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'stationid' in col_lower or 'automaticweatherstationid' in col_lower:
            id_col = col
        elif 'geometrylatitude' in col_lower or 'latitude' in col_lower or 'lat' in col_lower:
            lat_col = col
        elif 'geometrylongitude' in col_lower or 'longitude' in col_lower or 'lng' in col_lower or 'lon' in col_lower:
            lng_col = col
        elif 'stationname' in col_lower or 'automaticweatherstation_en' in col_lower or 'name' in col_lower:
            name_col = col  # 可选
    
    if None in (id_col, lat_col, lng_col):
        print(f"❌ 未找到所需列。现有列：{list(df.columns)}")
        return {}
    
    for idx, row in df.iterrows():
        sid = str(row[id_col]).strip()
        lat = float(row[lat_col])
        lng = float(row[lng_col])
        if sid and pd.notna(lat) and pd.notna(lng):
            name = row[name_col] if name_col else sid
            stations[sid] = {'name': name, 'lat': lat, 'lng': lng}
    
    print(f"✅ 成功读取 {len(stations)} 个气象站")
    return stations

# ===================== 6. 分配雨量到網格 =====================
def assign_rainfall_to_grid(grid, rainfall_dict, stations_dict):
    if not stations_dict:
        print("⚠️ 沒有站點數據，雨量設為 0")
        grid['rainfall'] = 0.0
        return grid
    stations_gdf = gpd.GeoDataFrame(
        [{'station_id': sid, 'geometry': Point(info['lng'], info['lat']), 'rainfall': rainfall_dict.get(sid, 0.0)}
         for sid, info in stations_dict.items()],
        crs='EPSG:4326'
    ).to_crs(epsg=2326)
    
    grid['rainfall'] = 0.0
    for idx, row in grid.iterrows():
        center = row.geometry.centroid
        distances = stations_gdf.geometry.distance(center)
        nearest_idx = distances.idxmin()
        grid.at[idx, 'rainfall'] = stations_gdf.loc[nearest_idx, 'rainfall']
    return grid

# ===================== 7. 計算風險指數 =====================
def normalize(series):
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return series * 0.0
    return (series - min_val) / (max_val - min_val)

def calculate_risk(grid):
    # 确保数据类型
    grid['elevation'] = grid['elevation'].astype(float)
    grid['rainfall'] = grid['rainfall'].astype(float)
    grid['inlet_count'] = grid['inlet_count'].astype(float)

    # ---- 百分位数排名归一化（0~1均匀分布） ----
    def rank_normalize(series):
        return series.rank(pct=True)   # 最小值→0，最大值→1，均匀分布

    grid['rainfall_factor'] = rank_normalize(grid['rainfall'])
    grid['inlet_factor'] = 1 - rank_normalize(grid['inlet_count'])
    grid['elevation_factor'] = 1 - rank_normalize(grid['elevation'])

    # ---- 动态权重 ----
    if grid['rainfall'].max() == 0:
        w_rain, w_inlet, w_elev = 0.0, 0.5, 0.5
        print("⚠️ 今日无雨，雨量权重归零，渠盖与地势各占50%")
    else:
        w_rain, w_inlet, w_elev = 0.3, 0.4, 0.3

    grid['risk_score'] = (grid['rainfall_factor'] * w_rain +
                          grid['inlet_factor'] * w_inlet +
                          grid['elevation_factor'] * w_elev) * 100

    grid['risk_score'] = grid['risk_score'].clip(0, 100)

    # 诊断信息
    print(f"  因子均值 - 雨量:{grid['rainfall_factor'].mean():.3f}, "
          f"渠盖:{grid['inlet_factor'].mean():.3f}, "
          f"地势:{grid['elevation_factor'].mean():.3f}")
    return grid

# ===================== 8. 主程式 =====================
def main():
    print("=" * 50)
    print("開始計算積水風險指數")
    print("=" * 50)
    
    # 1. 載入 DEM
    print("📂 載入 DEM...")
    try:
        dem_data, transform, bounds, crs = load_dem(DEM_PATH)
        print(f"✅ DEM 範圍：{bounds}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("請確認 DEM_PATH 變數指向正確的 GeoTIFF 檔案。")
        return
    
    # 2. 建立網格
    print("🔲 建立 500m 網格...")
    grid = create_grid(bounds, cell_size=500)
    print(f"✅ 建立 {len(grid)} 個網格")
    
    # 3. 提取海拔
    print("⛰️ 提取海拔...")
    grid = extract_elevation(grid, dem_data, transform)
    print(f"✅ 有效海拔網格：{len(grid)} 個")

    grid['geometry'] = grid['geometry'].simplify(tolerance=1.0)  # 单位：米，1米精度足够
    
    # 儲存海拔網格 (以便日後重用)
    grid.to_file("data/elevation_grid.geojson", driver="GeoJSON")
    print("✅ 已儲存 data/elevation_grid.geojson")
    
    # 4. 載入進水渠蓋密度 (修复版)
    print("🕳️ 載入進水渠蓋密度...")
    try:
        inlets = gpd.read_file("data/inlets_density.geojson")
        inlets_utm = inlets.to_crs(epsg=2326)
        # 关键修复：重置索引，避免重复标签报错
        grid_reset = grid.reset_index(drop=True)
        joined = gpd.sjoin(grid_reset, inlets_utm, how='left', predicate='intersects')
        # 按 joined 的索引分组，取第一个有效 count
        inlet_counts = joined.groupby(joined.index)['count'].first().fillna(0)
        grid['inlet_count'] = inlet_counts.values
        print(f"✅ 渠蓋密度合併完成")
    except Exception as e:
        print(f"⚠️ 載入渠蓋密度失敗：{e}，將使用 0 代替")
        grid['inlet_count'] = 0
    
    # 5. 獲取雨量數據
    print("🌧️ 獲取雨量數據...")
    rainfall = get_rainfall_data()
    stations = load_station_coords()
    grid = assign_rainfall_to_grid(grid, rainfall, stations)
    print(f"✅ 雨量分配完成")
    
    # 6. 計算風險
    print("📊 計算風險指數...")
    grid = calculate_risk(grid)
    print(f"✅ 風險分數範圍：{grid['risk_score'].min():.1f} - {grid['risk_score'].max():.1f}")
    
    # 7. 儲存結果 (轉為 WGS84)
    grid_wgs84 = grid.to_crs(epsg=4326)
    grid_wgs84.to_file("data/risk_zones.geojson", driver="GeoJSON")
    print("✅ 風險地圖已儲存：data/risk_zones.geojson")
    
    # 8. 輸出統計
    print("\n📈 風險統計：")
    print(grid['risk_score'].describe())

if __name__ == "__main__":
    main()