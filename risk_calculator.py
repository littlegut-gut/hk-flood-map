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
from io import StringIO
from datetime import datetime
from rasterstats import zonal_stats
import warnings
warnings.filterwarnings('ignore')

# ===================== 設定 =====================
# 請修改為你實際的 DEM 檔案路徑
DEM_PATH = r"D:\hk_flood_map\data\DEM\Digital Terrain Model.tif"

# ===================== 防洪設施座標（WGS84）=====================

# 地下蓄洪池（5 個）
STORAGE_TANKS = {
    "大坑東蓄洪池":     {"lat": 22.3265, "lng": 114.1660, "capacity": 100000},
    "上環蓄洪池":       {"lat": 22.2880, "lng": 114.1500, "capacity": 9380},
    "跑馬地地下蓄洪池": {"lat": 22.2729, "lng": 114.1824, "capacity": 60000},
    "安秀道蓄洪池":     {"lat": 22.3295, "lng": 114.2320, "capacity": 18000},
    "安達臣道蓄洪池":   {"lat": 22.3311, "lng": 114.2310, "capacity": 60000},
}

# 雨水排放隧道（4 條，簡化為起點/終點）
TUNNELS = {
    "港島西雨水排放隧道": {
        "start": (22.2719, 114.1649),
        "end":   (22.2625, 114.1278),
        "length_km": 10.5,
    },
    "荔枝角雨水排放隧道": {
        "start": (22.3415, 114.1509),
        "end":   (22.3290, 114.1449),
        "length_km": 3.7,
    },
    "荃灣雨水排放隧道": {
        "start": (22.3823, 114.1139),
        "end":   (22.3700, 114.1000),
        "length_km": 5.1,
    },
    "啟德雨水轉運計劃": {
        "start": (22.3370, 114.1790),
        "end":   (22.3300, 114.2000),
        "length_km": 1.5,
    },
}

# ===================== 私人管理區域（渠蓋/渠管數據缺失校正）=====================
PRIVATE_MANAGED_AREAS = {
    "香港國際機場": {
        "center": (22.3087, 113.9145),   # 赤鱲角中心
        "radius_m": 3500,                 # 覆蓋全島
    },
}

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

# ===================== 2. 建立 200m 網格 (HK1980) =====================
def create_grid(bounds, cell_size=200):
    xmin, ymin, xmax, ymax = bounds
    xs = np.arange(xmin, xmax, cell_size)
    ys = np.arange(ymin, ymax, cell_size)
    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]
    grid = gpd.GeoDataFrame({'geometry': cells}, crs='EPSG:2326')
    return grid

# ===================== 3. 提取海拔 =====================
def extract_elevation(grid, dem_data, transform):
    """向量化版本：一次計算所有網格的質心海拔。"""
    centroids = grid.geometry.centroid
    xs = centroids.x.values
    ys = centroids.y.values

    # rowcol 支援陣列輸入，一次返回所有行列
    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    valid = (
        (rows >= 0) & (rows < dem_data.shape[0]) &
        (cols >= 0) & (cols < dem_data.shape[1])
    )

    elevations = np.full(len(grid), np.nan)
    elevations[valid] = dem_data[rows[valid], cols[valid]]

    grid = grid.copy()
    grid['elevation'] = elevations
    grid = grid.dropna(subset=['elevation'])
    grid = grid[grid['elevation'] > 2.0]
    grid['elevation'] = grid['elevation'].astype(float)

    print(f"   ✅ 有效海拔網格：{len(grid)} 個（向量化）")
    return grid

# ===================== 3.5 土地用途 → 徑流係數 =====================
BLU_PATH = r"data/LUMHK_RasterGrid_2024/BLU.tif"

# BLU 兩位編碼 → 徑流係數 C 值
# ⚠️ 若你有官方元數據，請按官方分類修正此表
BLU_TO_RUNOFF_C = {
    # 住宅
    1:  0.75,   # 私人住宅
    2:  0.70,   # 公營房屋
    3:  0.55,   # 鄉郊居所

    # 商業
    11: 0.90,

    # 工業
    21: 0.90,   # 工業用地
    22: 0.90,   # 工業邨
    23: 0.85,   # 貨倉和露天貯物

    # 政府/機構/社區 + 休憩
    31: 0.70,   # G/IC
    32: 0.30,   # 休憩用地

    # 運輸
    41: 0.95,   # 道路和運輸設施
    42: 0.85,   # 鐵路
    43: 0.90,   # 機場
    44: 0.95,   # 港口設施

    # 未確定（保守中值）
    51: 0.50, 52: 0.50, 53: 0.50, 54: 0.50,

    # 水體
    61: 1.00,   # 水塘
    62: 1.00,   # 河道和明渠

    # 植被
    71: 0.15,   # 林地
    72: 0.20,   # 灌叢
    73: 0.25,   # 草地
    74: 0.10,   # 紅樹林及沼澤

    # 其他
    81: 0.50,   # 荒地
    83: 0.60,   # 岩岸
    91: 0.35,   # 農地
    92: 1.00,   # 魚塘
}

# 未匹配編碼時的保守默認值
DEFAULT_C = 0.5


def extract_runoff_coefficient(grid, blu_path=BLU_PATH):
    print(f"   當前映射表 keys：{sorted(BLU_TO_RUNOFF_C.keys())}")

    """
    用 zonal_stats 計算每個 200m 網格內的平均徑流係數。
    BLU 是分類柵格（categorical），先統計各類像素數，再按 C 值加權平均。
    """
    if not os.path.exists(blu_path):
        print(f"⚠️ 找不到 {blu_path}，runoff_coeff 設為 {DEFAULT_C}")
        grid['runoff_coeff'] = DEFAULT_C
        return grid

    try:
        with rasterio.open(blu_path) as src:
            print(f"   BLU CRS: {src.crs}，解析度: {src.res}")

        # categorical=True 回傳 {類別: 像素數} 的 dict
        stats = zonal_stats(
            grid.geometry,
            blu_path,
            categorical=True,
            nodata=0,             # 你的 BLU 用 0 表示 NoData
            all_touched=False,    # 只算質心落入的像素（較快）
        )

        runoff = []
        unknown_codes = set()
        for s in stats:
            if not s:
                runoff.append(np.nan)
                continue
            total = sum(s.values())
            if total == 0:
                runoff.append(np.nan)
                continue

            c_weighted = 0.0
            for cls, cnt in s.items():
                code = int(cls)
                if code not in BLU_TO_RUNOFF_C:
                    unknown_codes.add(code)
                c_val = BLU_TO_RUNOFF_C.get(code, DEFAULT_C)
                c_weighted += c_val * cnt
            runoff.append(c_weighted / total)

        grid['runoff_coeff'] = runoff

        # 未匹配值用陸地中值填補
        median_c = grid['runoff_coeff'].median()
        if pd.isna(median_c):
            median_c = DEFAULT_C
        grid['runoff_coeff'] = grid['runoff_coeff'].fillna(median_c)

        print(f"✅ 徑流係數計算完成")
        print(f"   範圍：{grid['runoff_coeff'].min():.3f} ~ "
              f"{grid['runoff_coeff'].max():.3f}，"
              f"平均 {grid['runoff_coeff'].mean():.3f}")

        if unknown_codes:
            print(f"   ⚠️ 未匹配的編碼：{sorted(unknown_codes)}（已用 {DEFAULT_C} 代替）")

        # 分佈診斷
        bins = pd.cut(grid['runoff_coeff'],
                      bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
                      labels=['低透水(<0.3)', '中低', '中', '中高', '高(>0.9)'])
        print(f"   分佈：{bins.value_counts().sort_index().to_dict()}")

    except Exception as e:
        print(f"⚠️ 計算徑流係數失敗：{e}")
        import traceback
        traceback.print_exc()
        grid['runoff_coeff'] = DEFAULT_C

    return grid


# ===================== 3.6 人口柵格 → 網格總人口 =====================
WORLDPOP_PATH = r"data/WorldPop/hkg_pop_2025_CN_100m_R2024B_v1.tif"

# ===================== 3.6a 預處理：重投影 WorldPop 到 EPSG:2326 =====================
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask as rio_mask
from shapely.geometry import box as shapely_box


def prepare_pop_raster(src_path, dst_path, dst_crs='EPSG:2326'):
    """
    把 WorldPop 從 EPSG:4326 重投影到 EPSG:2326，
    並裁剪到香港 DEM 覆蓋範圍（±1km buffer），大幅縮小檔案。
    """
    if os.path.exists(dst_path):
        print(f"   ✅ 已存在預處理柵格：{dst_path}")
        return dst_path

    # 香港 DEM 範圍（EPSG:2326），加 1km buffer
    HK_BOUNDS_2326 = shapely_box(
        799997.5 - 1000, 799997.5 - 1000,
        863752.5 + 1000, 848002.5 + 1000
    )

    try:
        with rasterio.open(src_path) as src:
            print(f"   原始柵格：{src.width} × {src.height}，CRS: {src.crs}")

            # 1. 先把香港範圍轉回源 CRS（EPSG:4326）
            from pyproj import Transformer
            transformer = Transformer.from_crs(dst_crs, src.crs, always_xy=True)
            # 取香港四角的經緯度
            xmin, ymin, xmax, ymax = HK_BOUNDS_2326.bounds
            lons, lats = [], []
            for x, y in [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]:
                lon, lat = transformer.transform(x, y)
                lons.append(lon)
                lats.append(lat)
            src_bounds_geom = shapely_box(min(lons) - 0.01, min(lats) - 0.01,
                                          max(lons) + 0.01, max(lats) + 0.01)

            # 2. 用香港範圍裁剪源柵格
            out_image, out_transform = rio_mask(
                src, [src_bounds_geom], crop=True, filled=True
            )
            out_meta = src.meta.copy()
            out_meta.update({
                'height': out_image.shape[1],
                'width':  out_image.shape[2],
                'transform': out_transform,
                'nodata': -99999.0,
            })

        # 3. 寫到臨時檔（EPSG:4326，香港範圍）
        tmp_path = dst_path + ".tmp.tif"
        with rasterio.open(tmp_path, 'w', **out_meta) as tmp:
            tmp.write(out_image)

        # 4. 重投影到 EPSG:2326
        with rasterio.open(tmp_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs,
                src.width, src.height, *src.bounds,
                resolution=100,        # 100m
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'nodata': -99999.0,
            })

            with rasterio.open(dst_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    dst_nodata=-99999.0,
                    resampling=Resampling.bilinear,
                )

        os.remove(tmp_path)
        with rasterio.open(dst_path) as dst:
            print(f"   ✅ 預處理完成：{dst.width} × {dst.height}，"
                  f"CRS: {dst.crs}，解析度: {dst.res}")
        return dst_path

    except Exception as e:
        print(f"   ⚠️ 預處理失敗：{e}")
        import traceback
        traceback.print_exc()
        return src_path


# ===================== 3.6 人口柵格 → 網格總人口 =====================
WORLDPOP_RAW = r"data/WorldPop/hkg_pop_2025_CN_100m_R2024B_v1.tif"
WORLDPOP_PREP = r"data/WorldPop/hkg_pop_2025_CN_100m_epsg2326.tif"


def extract_population_from_raster(grid, pop_path=WORLDPOP_PREP):
    """
    用 zonal_stats 計算每個 200m 網格內的總人口。
    WorldPop 100m 數據單位：每像素人數 → 用 sum 統計。
    """
    if not os.path.exists(pop_path):
        print(f"⚠️ 找不到 {pop_path}，人口總數設為 0")
        grid['population_total'] = 0.0
        grid['population_density'] = 0.0
        return grid

    try:
        with rasterio.open(pop_path) as src:
            print(f"   WorldPop CRS: {src.crs}，解析度: {src.res}")
            nodata_val = src.nodata

        stats = zonal_stats(
            grid.geometry,
            pop_path,
            stats=['sum'],
            nodata=nodata_val if nodata_val is not None else -99999,
            all_touched=False,
        )

        pop_values = [
            s['sum'] if s and s['sum'] is not None else 0.0
            for s in stats
        ]
        grid['population_total'] = pop_values

        grid.loc[grid['population_total'] < 0, 'population_total'] = 0.0
        grid['population_total'] = grid['population_total'].fillna(0.0)

        # 密度（人/km²）
        grid['population_density'] = grid['population_total'] / 0.04

        # ---- 高海拔去噪：海拔 > 600m 的網格一律視為無居住 ----
        # 依據：香港所有住宅區海拔 < 550m（太平山頂豪宅最高），
        #       600m 以上為純山頂/雷達站，WorldPop 在此處為模型噪聲
        high_mask = grid['elevation'] > 600
        n_high = high_mask.sum()
        if n_high > 0:
            grid.loc[high_mask, 'population_total'] = 0.0
            grid.loc[high_mask, 'population_density'] = 0.0
            print(f"   高海拔去噪：{n_high} 個網格（海拔>600m）人口歸零")

        print(f"✅ 人口柵格計算完成")
        print(f"   人口範圍：{grid['population_total'].min():.1f} ~ "
              f"{grid['population_total'].max():.0f} 人/網格，"
              f"平均 {grid['population_total'].mean():.1f}")
        print(f"   密度範圍：{grid['population_density'].min():.0f} ~ "
              f"{grid['population_density'].max():.0f} 人/km²，"
              f"平均 {grid['population_density'].mean():.0f}")

        zero_pop = (grid['population_total'] < 1).sum()
        print(f"   人口 < 1 的網格：{zero_pop} 個（{zero_pop/len(grid)*100:.1f}%）")

        # 抽樣檢查（用經緯度，自動轉投影）
        print(f"   --- 抽樣檢查 ---")
        from pyproj import Transformer
        transformer = Transformer.from_crs('EPSG:4326', 'EPSG:2326', always_xy=True)
        for name, lat, lng in [
            ("大帽山頂", 22.4106, 114.1241),   # 山頂雷達站
            ("旺角",     22.3193, 114.1694),
            ("米埔保護區", 22.4922, 114.0356),
            ("中環",     22.2819, 114.1583),
            ("西貢東郊野", 22.4060, 114.3380),
            ("太平山頂",  22.2716, 114.1500),
            ("天水圍",   22.4600, 114.0000),
            ("天水圍嘉湖", 22.4595, 114.0045),   # 嘉湖山莊核心
            ("中環住宅",  22.2830, 114.1520),   # 半山住宅區
        ]:
            try:
                x, y = transformer.transform(lng, lat)
                pt = Point(x, y)
                dists = grid.geometry.centroid.distance(pt)
                nearest_idx = dists.idxmin()
                row = grid.loc[nearest_idx]
                # 顯示距離，確認採樣點接近
                print(f"     {name} ({lat:.4f}, {lng:.4f}, 距最近網格 {dists.min():.0f}m): "
                      f"人口 {row['population_total']:.1f} 人/網格, "
                      f"密度 {row['population_density']:.0f} 人/km²")
            except Exception:
                pass

    except Exception as e:
        print(f"⚠️ 計算人口柵格失敗：{e}")
        import traceback
        traceback.print_exc()
        grid['population_total'] = 0.0
        grid['population_density'] = 0.0

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
    """
    用 IDW（反距離加權）將雨量插值到網格。
    - K=3：取最近 3 個氣象站
    - power=2：標準 IDW 衰減
    - 座標單位：米（EPSG:2326）
    """
    if not stations_dict:
        print("⚠️ 沒有站點數據，雨量設為 0")
        grid['rainfall'] = 0.0
        return grid

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("⚠️ 缺少 scipy，回退到最近鄰")
        return _assign_rainfall_nearest(grid, rainfall_dict, stations_dict)

    # ---- 建立站點 GeoDataFrame（EPSG:2326）----
    stations_gdf = gpd.GeoDataFrame(
        [{'station_id': sid,
          'geometry': Point(info['lng'], info['lat']),
          'rainfall': rainfall_dict.get(sid, 0.0)}
         for sid, info in stations_dict.items()],
        crs='EPSG:4326'
    ).to_crs(epsg=2326)

    # ---- 站點座標 + 雨量 ----
    station_xy = np.column_stack([
        stations_gdf.geometry.x.values,
        stations_gdf.geometry.y.values,
    ]).astype(float)
    station_rain = stations_gdf['rainfall'].values.astype(float)

    # ---- 網格質心座標（一次向量化）----
    centroids = grid.geometry.centroid
    grid_xy = np.column_stack([
        centroids.x.values,
        centroids.y.values,
    ]).astype(float)

    # ---- KDTree 查詢最近 K 個站 ----
    K = 3
    tree = cKDTree(station_xy)
    dists, idxs = tree.query(grid_xy, k=K)
    # dists: (N, K), idxs: (N, K)

    # ---- IDW 權重 ----
    EPS = 1.0    # 1 米，避免距離為 0 時除零
    weights = 1.0 / (dists + EPS) ** 2
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights = weights / np.maximum(weights_sum, 1e-12)

    # ---- 加權平均 ----
    rainfall_interp = (station_rain[idxs] * weights).sum(axis=1)
    grid['rainfall'] = rainfall_interp

    # ---- 診斷 ----
    print(f"✅ IDW 雨量插值完成（K={K}, power=2）")
    print(f"   雨量範圍：{grid['rainfall'].min():.2f} ~ "
          f"{grid['rainfall'].max():.2f} mm")
    print(f"   平均：{grid['rainfall'].mean():.2f} mm，"
          f"標準差：{grid['rainfall'].std():.2f} mm")

    # 與最近鄰對比
    nearest_dists, nearest_idx = tree.query(grid_xy, k=1)
    nearest_rain = station_rain[nearest_idx]
    diff = np.abs(rainfall_interp - nearest_rain)
    print(f"   與最近鄰的平均差異：{diff.mean():.2f} mm，"
          f"最大差異：{diff.max():.2f} mm")

    # 非零站數量
    n_nonzero = int((station_rain > 0).sum())
    print(f"   有雨站點：{n_nonzero}/{len(station_rain)}")

    return grid

def _assign_rainfall_nearest(grid, rainfall_dict, stations_dict):
    """備用方案：最近鄰（若 scipy 不可用）。"""
    stations_gdf = gpd.GeoDataFrame(
        [{'station_id': sid,
          'geometry': Point(info['lng'], info['lat']),
          'rainfall': rainfall_dict.get(sid, 0.0)}
         for sid, info in stations_dict.items()],
        crs='EPSG:4326'
    ).to_crs(epsg=2326)

    grid_points = grid.copy()
    grid_points['geometry'] = grid_points.geometry.centroid

    joined = gpd.sjoin_nearest(
        grid_points[['geometry']],
        stations_gdf[['rainfall', 'geometry']],
        how='left',
        distance_col='_dist_rain'
    )
    rainfall_series = joined.groupby(joined.index)['rainfall'].first()
    grid['rainfall'] = rainfall_series.reindex(grid.index).fillna(0.0)

    print(f"✅ 最近鄰雨量分配完成（備用）")
    print(f"   雨量範圍：{grid['rainfall'].min():.2f} ~ "
          f"{grid['rainfall'].max():.2f} mm")
    return grid
    
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
    for col in ['elevation', 'rainfall', 'inlet_count', 'dist_to_pipe',
                'population_total',                # ← 改為 population_total
                'dist_to_flood', 'coastal_factor', 'runoff_coeff']:
        if col in grid.columns:
            grid[col] = grid[col].astype(float)

    def rank_normalize(series):
        return series.rank(pct=True)

    # ---- 私人管理區域：渠蓋/渠管數據缺失校正 ----
    from shapely.geometry import Point
    from pyproj import Transformer

    transformer_private = Transformer.from_crs('EPSG:4326', 'EPSG:2326', always_xy=True)

    private_mask = pd.Series(False, index=grid.index)
    centroids_private = grid.geometry.centroid

    for name, info in PRIVATE_MANAGED_AREAS.items():
        x, y = transformer_private.transform(info['center'][1], info['center'][0])
        pt = Point(x, y)
        mask = (centroids_private.distance(pt) < info['radius_m'])
        private_mask |= mask
        n = int(mask.sum())
        if n > 0:
            print(f"     {name}：{n} 個網格（數據缺失校正）")

    if private_mask.sum() > 0:
        # 用「高密度市區」網格的中位數填充（人口 > 1000）
        # 這些網格與機場的營運密度最接近
        dense_urban = (~private_mask) & (grid['population_total'] > 1000)
        n_dense = int(dense_urban.sum())

        if n_dense > 100:
            median_inlet = grid.loc[dense_urban, 'inlet_count'].median()
            median_pipe  = grid.loc[dense_urban, 'dist_to_pipe'].median()
            print(f"   使用 {n_dense} 個高密度市區網格的中位數")
        else:
            # 回退到有人口的非私人區域
            active_non_private = (~private_mask) & (grid['population_total'] >= 1)
            median_inlet = grid.loc[active_non_private, 'inlet_count'].median()
            median_pipe  = grid.loc[active_non_private, 'dist_to_pipe'].median()
            print(f"   回退：使用 {int(active_non_private.sum())} 個活躍網格的中位數")

        grid.loc[private_mask, 'inlet_count'] = median_inlet
        grid.loc[private_mask, 'dist_to_pipe'] = median_pipe
        print(f"   私人管理區域填充：渠蓋 = {median_inlet:.1f}，"
              f"距渠管 = {median_pipe:.0f}m")

    # ---- 基础因子 ----
    # 降雨：改用固定归一化（50mm/h ≈ 黑雨级别），让前端可重算
    RAINFALL_NORM_MAX = 50.0
    grid['rainfall_factor'] = np.clip(grid['rainfall'] / RAINFALL_NORM_MAX, 0, 1)
    grid['inlet_factor'] = 1 - rank_normalize(grid['inlet_count'])
    grid['elevation_factor'] = 1 - rank_normalize(grid['elevation'])
    grid['pipe_factor'] = rank_normalize(grid['dist_to_pipe'])
    grid['population_factor'] = rank_normalize(grid['population_total'])
   
    # ---- 郊野屏蔽：人口 = 0 → 渠蓋/渠管歸零 ----
    is_rural = grid['population_total'] < 1
    n_rural = is_rural.sum()
    grid.loc[is_rural, 'inlet_factor'] = 0.0
    grid.loc[is_rural, 'pipe_factor']  = 0.0

    # ---- 郊野公園屏蔽：runoff_coeff 強制 = 0.10 ----  ← 新增
    if 'in_park' in grid.columns:
        park_mask = grid['in_park'] == True
        n_park = park_mask.sum()
        if n_park > 0:
            grid.loc[park_mask, 'runoff_coeff'] = 0.10
            print(f"   郊野公園屏蔽：{n_park} 個網格 runoff_coeff 強制設為 0.10")

    # ---- 重新計算 runoff_factor（因為 runoff_coeff 已被修改）----
    grid['runoff_factor'] = rank_normalize(grid['runoff_coeff'])

    print(f"   郊野屏蔽：{n_rural} 個網格（人口=0）渠蓋/渠管因子歸零")

    # ---- 水體屏蔽：疑似水庫 → 土地因子歸零 ----
    is_reservoir = (
        (grid['population_total'] < 1) &
        (grid['runoff_coeff'].between(0.30, 0.50)) &
        (grid['elevation'].between(30, 60))
    )
    n_reservoir = is_reservoir.sum()
    if n_reservoir > 0:
        grid.loc[is_reservoir, 'runoff_factor'] = 0.0
        print(f"   水體屏蔽：{n_reservoir} 個網格（疑似水庫）土地因子歸零")

    # ---- 历史积水点因子（指数衰减）----
    scale = 2000.0
    grid['flood_factor'] = np.exp(-grid['dist_to_flood'] / scale)

    # ---- 沿海脆弱性因子（已是 0~1）----
    grid['coastal_factor'] = grid['coastal_factor'].clip(0, 1)

    # ---- 动态权重 ----
    max_rain = grid['rainfall'].max()
    max_surge = grid['storm_surge'].max() if 'storm_surge' in grid.columns else 0
    has_storm_surge = max_surge > 0.3

    if max_rain == 0 and not has_storm_surge:
        w = {'rain': 0.00, 'inlet': 0.12, 'elev': 0.13, 'pipe': 0.20,
             'pop': 0.25, 'coastal': 0.13, 'runoff': 0.17}
        print("⚠️ 無雨無風暴潮，權重：渠蓋0.12 地勢0.13 渠管0.20 人口0.25 沿海0.13 土地0.17")
    elif max_rain == 0 and has_storm_surge:
        w = {'rain': 0.00, 'inlet': 0.08, 'elev': 0.08, 'pipe': 0.12,
             'pop': 0.12, 'coastal': 0.45, 'runoff': 0.15}
        print(f"🌊 有風暴潮（增水{max_surge:.1f}m），權重：沿海0.45 土地0.15")
    elif max_rain > 0 and has_storm_surge:
        w = {'rain': 0.20, 'inlet': 0.08, 'elev': 0.08, 'pipe': 0.12,
             'pop': 0.08, 'coastal': 0.30, 'runoff': 0.14}
        print(f"⛈️ 暴雨+風暴潮（增水{max_surge:.1f}m），權重：雨量0.20 沿海0.30 土地0.14")
    else:
        w = {'rain': 0.22, 'inlet': 0.12, 'elev': 0.13, 'pipe': 0.17,
             'pop': 0.19, 'coastal': 0.00, 'runoff': 0.17}
        print("🌧️ 有雨無風暴潮，權重：雨量0.22 渠蓋0.12 地勢0.13 渠管0.17 人口0.19 土地0.17")

    # ---- 不含降雨的基座分數 ----
    base_score_no_rain = (
        grid['inlet_factor']      * w['inlet'] +
        grid['elevation_factor']  * w['elev'] +
        grid['pipe_factor']       * w['pipe'] +
        grid['population_factor'] * w['pop'] +
        grid['coastal_factor']    * w['coastal'] +
        grid['runoff_factor']     * w['runoff']
    ) * 100

    # ---- 历史积水点额外惩罚 ----
    flood_penalty = grid['flood_factor'] * 30.0

    # 基座分（不含降雨，含历史惩罚）—— 输出给前端用
    grid['base_score'] = (base_score_no_rain + flood_penalty).clip(0, 100)

    # 最终风险分（含降雨）
    grid['risk_score'] = (
        grid['base_score'] + grid['rainfall_factor'] * w['rain'] * 100
    ).clip(0, 100)

    # 把降雨权重也存起来（供前端参考）
    grid['rainfall_weight'] = w['rain']

    # ============================================================
    # 防洪設施風險下調（策略 B）—— 必須放在 risk_score 創建之後
    # ============================================================
    print("   計算防洪設施風險下調...")

    from shapely.geometry import Point, LineString
    from pyproj import Transformer

    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:2326', always_xy=True)

    STORAGE_BUFFER = 800
    TUNNEL_BUFFER  = 500
    RISK_REDUCTION = 0.85

    # 用 pd.Series 確保 index 與 grid 對齊
    facility_mask = pd.Series(False, index=grid.index)
    centroids = grid.geometry.centroid

    # 1. 蓄洪池
    for name, info in STORAGE_TANKS.items():
        x, y = transformer.transform(info['lng'], info['lat'])
        pt = Point(x, y)
        mask = (centroids.distance(pt) < STORAGE_BUFFER)
        facility_mask |= mask
        n = int(mask.sum())
        if n > 0:
            print(f"     {name}：{n} 個網格受惠")

    # 2. 隧道
    for name, info in TUNNELS.items():
        sx, sy = transformer.transform(info['start'][1], info['start'][0])
        ex, ey = transformer.transform(info['end'][1], info['end'][0])
        line = LineString([(sx, sy), (ex, ey)])
        mask = (centroids.distance(line) < TUNNEL_BUFFER)
        facility_mask |= mask
        n = int(mask.sum())
        if n > 0:
            print(f"     {name}：{n} 個網格受惠")

    n_facility = int(facility_mask.sum())
    if n_facility > 0:
        grid.loc[facility_mask, 'risk_score'] *= RISK_REDUCTION
        print(f"   防洪設施屏蔽：{n_facility} 個網格風險下調 15%")
    else:
        print("   ℹ️ 無網格落在防洪設施服務範圍內")

    # ---- 诊断（无条件打印）----
    print(f"  人口統計 - 總人口均值:{grid['population_total'].mean():.1f} 人/網格, "
          f"零人口網格數:{(grid['population_total'] < 1).sum()}")
    print(f"  因子均值 - 雨量:{grid['rainfall_factor'].mean():.3f}, "
          f"渠盖:{grid['inlet_factor'].mean():.3f}, "
          f"地势:{grid['elevation_factor'].mean():.3f}, "
          f"渠管:{grid['pipe_factor'].mean():.3f}, "
          f"人口:{grid['population_factor'].mean():.3f}, "
          f"沿海:{grid['coastal_factor'].mean():.3f}, "
          f"土地:{grid['runoff_factor'].mean():.3f}, "
          f"歷史積水:{grid['flood_factor'].mean():.3f}")
    print(f"  歷史積水懲罰均值: {flood_penalty.mean():.2f} 分")

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
    print("🔲 建立 200m 網格...")
    grid = create_grid(bounds, cell_size=200)
    print(f"✅ 建立 {len(grid)} 個網格")
    
    # 3. 提取海拔
    print("⛰️ 提取海拔...")
    grid = extract_elevation(grid, dem_data, transform)
    print(f"✅ 有效海拔網格：{len(grid)} 個")

    grid['geometry'] = grid['geometry'].simplify(tolerance=1.0)  # 单位：米，1米精度足够
    
    # 儲存海拔網格 (以便日後重用)
    grid.to_file("data/elevation_grid.geojson", driver="GeoJSON")
    print("✅ 已儲存 data/elevation_grid.geojson")
    
    # 3.5 提取徑流係數
    print("🏙️ 提取土地利用徑流係數...")
    grid = extract_runoff_coefficient(grid)

    grid['geometry'] = grid['geometry'].simplify(tolerance=1.0)

    # ===== 3.55 載入郊野公園邊界 =====
    print("🏞️ 載入郊野公園邊界...")
    AFCD_PATH = "data/afcd_country_parks.geojson"

    if os.path.exists(AFCD_PATH):
        try:
            afcd = gpd.read_file(AFCD_PATH)
            if afcd.crs is None:
                afcd = afcd.set_crs(epsg=4326, allow_override=True)
            afcd_utm = afcd.to_crs(epsg=2326)

            # 只保留 Polygon / MultiPolygon，過濾掉 LineString
            afcd_utm = afcd_utm[
                afcd_utm.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
            ].copy()

            # ===== 新增：過濾非郊野公園 feature =====
            name_col = None
            for cand in ['name:zh', 'name', 'name:en']:
                if cand in afcd_utm.columns:
                    name_col = cand
                    break

            if name_col:
                names = afcd_utm[name_col].astype(str)
                # 黑名單：海岸公園、海域、深圳公園、維港、白海豚、桂山島
                EXCLUDE = (
                    '海岸|Marine|海豚|維多利亞港|桂山|梅林山|塘朗山|深圳|珠江口'
                )
                mask_exclude = names.str.contains(EXCLUDE, case=False, na=False)
                before = len(afcd_utm)
                afcd_utm = afcd_utm[~mask_exclude].copy()
                print(f"   過濾非郊野公園 feature：{before} → {len(afcd_utm)} 個")

            # 修復自相交 + 過濾面積過小的碎片
            afcd_utm['geometry'] = afcd_utm.geometry.buffer(0)
            afcd_utm = afcd_utm[afcd_utm.geometry.area > 10000]
            print(f"   有效邊界：{len(afcd_utm)} 個")

            # ===== 新增：過濾掉海岸公園／海岸保護區 =====
            # 這些是海域邊界，不應用於陸地 C 值屏蔽
            name_col = None
            for cand in ['name:zh', 'name', 'name:en']:
                if cand in afcd_utm.columns:
                    name_col = cand
                    break

            if name_col:
                before = len(afcd_utm)
                marine_mask = afcd_utm[name_col].astype(str).str.contains(
                    '海岸公園|海岸保護區|Marine Park|Marine Reserve',
                    case=False, na=False
                )
                afcd_utm = afcd_utm[~marine_mask].copy()
                print(f"   過濾海岸公園：{before} → {len(afcd_utm)} 個邊界")

            # 修復自相交 + 過濾面積過小的碎片
            afcd_utm['geometry'] = afcd_utm.geometry.buffer(0)
            afcd_utm = afcd_utm[afcd_utm.geometry.area > 10000]

            print(f"   有效多邊形邊界：{len(afcd_utm)} 個")

            # 空間連接：用 intersects 捕捉邊界網格
            joined = gpd.sjoin(
                grid[['geometry']],
                afcd_utm[['geometry']],
                how='left',
                predicate='intersects'
            )
            # 修正：左連接保留所有網格，必須檢查 index_right 是否非空
            matched = joined[joined['index_right'].notna()]
            matched_ids = matched.index.unique()
            grid['in_park'] = grid.index.isin(matched_ids)

            n_park = grid['in_park'].sum()
            print(f"   ✅ 標記 {n_park} 個網格位於保護區範圍內 "
                f"（{n_park/len(grid)*100:.1f}%）")
            
            n_park = grid['in_park'].sum()
            print(f"   ✅ 標記 {n_park} 個網格位於保護區範圍內 "
                f"（{n_park/len(grid)*100:.1f}%）")
        except Exception as e:
            print(f"   ⚠️ 郊野公園邊界處理失敗：{e}")
            import traceback; traceback.print_exc()
            grid['in_park'] = False
    else:
        grid['in_park'] = False
        print(f"   ℹ️ 找不到 {AFCD_PATH}，in_park 全部設為 False")

      # 3.6 提取人口柵格（先預處理重投影）
    print("👥 提取人口柵格...")
    print("   🔧 預處理 WorldPop 柵格（重投影 + 裁剪）...")
    prepare_pop_raster(WORLDPOP_RAW, WORLDPOP_PREP)
    grid = extract_population_from_raster(grid)

    # ===== 4. 載入雨水進水口格柵 (Drain Gully Grating) 作為渠蓋密度 =====
    print("🕳️ 載入雨水進水口格柵 (Drain Gully Grating)...")
    try:
        gully = gpd.read_file("data/Drain_Gully_Grating_SHP/Drain_Gully_Grating_SHP.shp")
        if gully.crs is None:
            gully = gully.set_crs(epsg=2326, allow_override=True)
        else:
            gully = gully.to_crs(epsg=2326)
        print(f"✅ 載入 {len(gully)} 個進水口格柵")
        
        # 直接用 grid（不 reset_index），保持索引一致
        joined = gpd.sjoin(grid[['geometry']], gully[['geometry']],
                          how='left', predicate='intersects')
        # 只統計真正匹配的（index_right 非 NaN）
        matched = joined[joined['index_right'].notna()]
        inlet_counts = matched.groupby(matched.index).size()
        grid['inlet_count'] = inlet_counts.reindex(grid.index).fillna(0).astype(int)
        print(f"✅ 渠蓋密度計算完成，平均每網格 {grid['inlet_count'].mean():.2f} 個進水口")
    except Exception as e:
        print(f"⚠️ 載入進水口格柵失敗：{e}，將使用 0 代替")
        grid['inlet_count'] = 0
    

    # ===== 4.5 合併兩個雨水渠管 SHP 並計算距離 =====
    print("🔧 合併雨水渠管圖層...")
    
    pipe_layers = []
    
    try:
        pipe1 = gpd.read_file("data/Pipe_Stormwater_SHP/Pipe_Stormwater_SHP.shp")
        if pipe1.crs is None:
            pipe1 = pipe1.set_crs(epsg=2326, allow_override=True)
        else:
            pipe1 = pipe1.to_crs(epsg=2326)
        pipe_layers.append(pipe1)
        print(f"✅ 載入 Pipe_Stormwater_SHP：{len(pipe1)} 條")
    except Exception as e:
        print(f"⚠️ 載入 Pipe_Stormwater_SHP 失敗：{e}")
    
    try:
        pipe2 = gpd.read_file("data/Multiple_Pipes_Stormwater/Multiple_Pipes_Stormwater.shp")
        if pipe2.crs is None:
            pipe2 = pipe2.set_crs(epsg=2326, allow_override=True)
        else:
            pipe2 = pipe2.to_crs(epsg=2326)
        pipe_layers.append(pipe2)
        print(f"✅ 載入 Multiple_Pipes_Stormwater：{len(pipe2)} 條")
    except Exception as e:
        print(f"⚠️ 載入 Multiple_Pipes_Stormwater 失敗：{e}")
    
    if pipe_layers:
        merged_pipes = gpd.GeoDataFrame(
            pd.concat([layer[['geometry']] for layer in pipe_layers], ignore_index=True),
            crs='EPSG:2326'
        )
        print(f"✅ 合併後共有 {len(merged_pipes)} 條雨水渠管")
        
        merged_pipes['geometry'] = merged_pipes.geometry.simplify(2.0, preserve_topology=True)
        print(f"✅ 渠管已簡化")
        
        # ===== 使用 sjoin_nearest 计算最近距离（推荐）=====
        # 将网格质心作为点
        grid_points = grid.copy()
        grid_points['geometry'] = grid_points.geometry.centroid
        
        # 最近邻空间连接，distance_col 自动生成距离列
        joined = gpd.sjoin_nearest(
            grid_points[['geometry']],
            merged_pipes[['geometry']],
            how='left',
            distance_col='dist_to_pipe'
        )
        # 由于索引可能重复（一个网格可能匹配多条等距管线），取每个网格的第一个
        dist_series = joined.groupby(joined.index)['dist_to_pipe'].first()
        grid['dist_to_pipe'] = dist_series.reindex(grid.index).fillna(99999.0)
        
        print(f"✅ 渠管距離計算完成")
        print(f"   距離統計：最小 {grid['dist_to_pipe'].min():.1f}m，"
              f"最大 {grid['dist_to_pipe'].max():.1f}m，"
              f"平均 {grid['dist_to_pipe'].mean():.1f}m")
    else:
        print("❌ 沒有任何渠管圖層被成功載入，將使用默認距離 0")
        grid['dist_to_pipe'] = 0.0


    # ===== 4.65 載入潮汐與風暴潮數據 =====
    print("🌊 載入潮汐與風暴潮數據...")

    # 先給默認值，確保後面無論如何都有欄位可用
    grid['nearest_tide_station'] = ''
    grid['dist_to_tide']        = 99999.0
    grid['real_time_tide']      = 0.0
    grid['storm_surge']         = 0.0

    ASTRONOMICAL_HIGH_TIDE = 2.0  # 天文高潮典型值（米）

    try:
        # --- 1. 讀取潮汐站 SHP ---
        tide_stations = gpd.read_file(
            "data/Predicted_tidal_information_Times_and_heights_SHP/HLT_converted.shp"
        )
        if tide_stations.crs is None:
            tide_stations = tide_stations.set_crs(epsg=4326, allow_override=True)
        tide_stations_utm = tide_stations.to_crs(epsg=2326)
        print(f"✅ 載入 {len(tide_stations_utm)} 個潮汐站")

        # --- 2. 英文站名 → 中文名映射 ---
        name_mapping = {
            'Quarry Bay': '鰂魚涌', 'Shek Pik': '石壁', 'Tai O': '大澳',
            'Tsim Bei Tsui': '尖鼻咀', 'Tai Miu Wan': '大廟灣',
            'Tai Po Kau': '大埔滘', 'Waglan Island': '橫瀾島',
            # 無實時數據 → 映射到最近有數據的站
            'Cheung Chau': '石壁', 'Chek Lap Kok (E)': '尖鼻咀',
            'Chi Ma Wan': '石壁', 'Kwai Chung': '鰂魚涌',
            'Ko Lau Wan': '大埔滘', 'Lok On Pai': '尖鼻咀',
            'Ma Wan': '大廟灣', 'Po Toi': '橫瀾島',
        }

        # --- 3. 讀取實時潮汐 CSV ---
        real_tide = {}
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
            print(f"✅ 讀取實時潮汐 {len(real_tide)} 個站：{list(real_tide.keys())}")
        except Exception as e:
            print(f"⚠️ 無法獲取實時潮汐數據：{e}")

        # --- 4. 過濾出有實時數據的站 ---
        tide_stations_utm['_zh_name'] = tide_stations_utm['TideStatio'].map(name_mapping)
        stations_with_data = tide_stations_utm[
            tide_stations_utm['_zh_name'].isin(real_tide.keys())
        ].copy()
        print(f"✅ 有實時數據的潮汐站：{len(stations_with_data)} 個")

        # --- 5. 空間連接（有數據才做）---
        if len(stations_with_data) > 0 and len(real_tide) > 0:
            grid_points = grid.copy()
            grid_points['geometry'] = grid_points.geometry.centroid

            joined = gpd.sjoin_nearest(
                grid_points[['geometry']],
                stations_with_data[['_zh_name', 'geometry']],
                how='left',
                distance_col='dist_to_tide'
            )
            grid['nearest_tide_station'] = (
                joined.groupby(joined.index)['_zh_name'].first()
                .reindex(grid.index).fillna('')
            )
            grid['dist_to_tide'] = (
                joined.groupby(joined.index)['dist_to_tide'].first()
                .reindex(grid.index).fillna(99999.0)
            )
            grid['real_time_tide'] = (
                grid['nearest_tide_station'].map(real_tide).fillna(0.0)
            )
            print(f"   平均距最近潮汐站：{grid['dist_to_tide'].mean():.0f} m")
        else:
            print("⚠️ 無可用潮汐站/實時數據，維持默認值 0")

        # --- 6. 風暴潮增水（無論如何都算一次，因為 real_time_tide 一定有值）---
        grid['storm_surge'] = (
            grid['real_time_tide'] - ASTRONOMICAL_HIGH_TIDE
        ).clip(lower=0)

        print(f"✅ 潮位範圍：{grid['real_time_tide'].min():.2f} ~ "
              f"{grid['real_time_tide'].max():.2f} m")
        print(f"✅ 風暴潮增水範圍：{grid['storm_surge'].min():.2f} ~ "
              f"{grid['storm_surge'].max():.2f} m")
        if grid['storm_surge'].max() == 0:
            print("   ℹ️ 當前無風暴潮（潮位未超過典型高潮位）")

    except Exception as e:
        print(f"⚠️ 載入潮汐數據失敗：{e}")
        import traceback
        traceback.print_exc()
        # 已在開頭給過默認值，這裡無需再賦值
        # storm_surge 保持 0.0，不影響後續計算

    # ===== 4.7 載入歷史積水黑點 =====
    print("💧 載入歷史積水黑點...")
    try:
        flood = gpd.read_file("data/FloodingBlackspots.geojson")
        # 统一坐标系
        if flood.crs is None:
            flood = flood.set_crs(epsg=4326, allow_override=True)
        flood = flood.to_crs(epsg=2326)
        print(f"✅ 載入 {len(flood)} 個歷史積水黑點")
        
        # 取網格質心
        grid_points = grid.copy()
        grid_points['geometry'] = grid_points.geometry.centroid
        
        # 最近邻空间连接
        joined = gpd.sjoin_nearest(
            grid_points[['geometry']],
            flood[['geometry']],
            how='left',
            distance_col='dist_to_flood'
        )
        dist_series = joined.groupby(joined.index)['dist_to_flood'].first()
        grid['dist_to_flood'] = dist_series.reindex(grid.index).fillna(99999.0)
        
        print(f"✅ 歷史積水點距離計算完成")
        print(f"   距離統計：最小 {grid['dist_to_flood'].min():.1f}m，"
              f"最大 {grid['dist_to_flood'].max():.1f}m，"
              f"平均 {grid['dist_to_flood'].mean():.1f}m")
    except Exception as e:
        print(f"⚠️ 載入歷史積水點失敗：{e}，將使用默認距離 99999")
        grid['dist_to_flood'] = 99999.0
   
    # 4.8. 获取实时风暴潮增水（使用 CSV API）
    
    # ===== 4.75 計算沿海脆弱性因子（擴大覆蓋版）=====
    print("🏖️ 計算沿海脆弱性...")
    try:
        # --- 1. 標記官方風險區（若存在）---
        try:
            coastal_hotspots = gpd.read_file("data/coastal_hotspots.shp")
            if coastal_hotspots.crs is None:
                coastal_hotspots = coastal_hotspots.set_crs(epsg=2326, allow_override=True)
            else:
                coastal_hotspots = coastal_hotspots.to_crs(epsg=2326)
            
            grid_reset = grid.reset_index(drop=True)
            joined_hot = gpd.sjoin(grid_reset, coastal_hotspots[['geometry']], 
                                    how='left', predicate='intersects')
            hotspot_flag = joined_hot.groupby(joined_hot.index).size() > 0
            grid['is_coastal_hotspot'] = hotspot_flag.reindex(grid.index).fillna(False).astype(int)
            print(f"✅ 標記 {grid['is_coastal_hotspot'].sum()} 個網格為官方沿海風險區")
        except Exception:
            print("ℹ️ 找不到 coastal_hotspots.shp，使用地形+潮位判斷")
            grid['is_coastal_hotspot'] = 0
        
        # --- 2. 計算沿海脆弱性（擴大覆蓋範圍）---
        def calculate_coastal_vulnerability(row):
            """
            成分A：永久地形脆弱性
              - 海拔 < 20m 且距海 < 10km 的區域有基礎脆弱性
            成分B：動態風暴潮加成（潮位 > 2.0m 時激活）
            """
            elev = row['elevation']
            dist = row['dist_to_tide']
            tide = row['real_time_tide']
            
            # ===== 成分A：永久地形脆弱性 =====
            # 海拔因子：20m 以下開始有脆弱性
            if elev < 20:
                elev_factor = (20 - elev) / 20.0  # 0m→1.0, 20m→0.0
            else:
                elev_factor = 0.0
            
            # 距離因子：10km 內有效
            dist_factor = np.exp(-dist / 10000.0)
            
            permanent_vuln = elev_factor * dist_factor
            
            # ===== 成分B：動態風暴潮加成 =====
            ASTRONOMICAL_HIGH = 2.0
            if tide > ASTRONOMICAL_HIGH:
                surge = tide - ASTRONOMICAL_HIGH
                surge_factor = min(1.0, surge / 2.0)
                surge_dist_factor = np.exp(-dist / 3000.0)
                surge_vuln = surge_factor * surge_dist_factor
            else:
                surge_vuln = 0.0
            
            # ===== 合併 =====
            vulnerability = min(1.0, permanent_vuln + surge_vuln)
            
            if row['is_coastal_hotspot'] == 1:
                vulnerability = min(1.0, vulnerability + 0.3)
            
            return vulnerability
        def calculate_coastal_vulnerability_vectorized(grid):
            """向量化計算沿海脆弱性。"""
            elev    = grid['elevation'].values.astype(float)
            dist    = grid['dist_to_tide'].values.astype(float)
            tide    = grid['real_time_tide'].values.astype(float)
            hotspot = grid['is_coastal_hotspot'].values.astype(int)

            # ===== 成分 A：永久地形脆弱性 =====
            elev_factor = np.where(elev < 20, (20.0 - elev) / 20.0, 0.0)
            elev_factor = np.clip(elev_factor, 0.0, 1.0)
            dist_factor = np.exp(-dist / 10000.0)
            permanent_vuln = elev_factor * dist_factor

            # ===== 成分 B：動態風暴潮加成 =====
            surge = np.maximum(0.0, tide - 2.0)
            surge_factor = np.minimum(1.0, surge / 2.0)
            surge_dist_factor = np.exp(-dist / 3000.0)
            surge_vuln = surge_factor * surge_dist_factor

            # ===== 合併 =====
            vuln = np.minimum(1.0, permanent_vuln + surge_vuln)
            vuln = np.where(hotspot == 1, np.minimum(1.0, vuln + 0.3), vuln)

            return vuln

        grid['coastal_factor'] = calculate_coastal_vulnerability_vectorized(grid)
        print(f"✅ 沿海脆弱性因子計算完成")
        print(f"   因子均值：{grid['coastal_factor'].mean():.3f}，"
              f"最大值：{grid['coastal_factor'].max():.3f}，"
              f"非零網格數：{(grid['coastal_factor'] > 0).sum()}")
        
        low_elev = (grid['elevation'] < 5).sum()
        mid_elev = (grid['elevation'] < 20).sum()
        near_coast = (grid['dist_to_tide'] < 2000).sum()
        print(f"   診斷：海拔<5m：{low_elev}，海拔<20m：{mid_elev}，距站<2km：{near_coast}")
        
    except Exception as e:
        print(f"⚠️ 沿海脆弱性計算失敗：{e}")
        import traceback
        traceback.print_exc()
        grid['coastal_factor'] = 0.0

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