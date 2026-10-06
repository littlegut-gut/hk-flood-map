# aggregate_inlets.py (已更新為你的資料夾名稱)
import os
import glob
import geopandas as gpd
import numpy as np
from shapely.geometry import box

def find_shp(folder):
    files = glob.glob(os.path.join(folder, "*.shp"))
    if not files:
        raise FileNotFoundError(f"在 {folder} 中找不到 SHP 檔案")
    return files[0]

def aggregate_inlets(input_shp, output_geojson, grid_size=500):
    print("📂 讀取進水渠蓋數據...")
    gdf = gpd.read_file(input_shp)
    print(f"✅ 原始點數：{len(gdf)}")

    # 轉為 HK1980 Grid (EPSG:2326) 計算距離
    gdf_utm = gdf.to_crs(epsg=2326)

    xmin, ymin, xmax, ymax = gdf_utm.total_bounds
    xs = np.arange(xmin, xmax, grid_size)
    ys = np.arange(ymin, ymax, grid_size)

    cells = [box(x, y, x + grid_size, y + grid_size) for x in xs for y in ys]
    grid = gpd.GeoDataFrame({'geometry': cells}, crs='EPSG:2326')

    joined = gpd.sjoin(gdf_utm, grid, how='left', predicate='within')
    counts = joined.groupby('index_right').size().reset_index(name='count')
    grid['count'] = 0
    grid.loc[counts['index_right'], 'count'] = counts['count'].values

    grid_wgs84 = grid.to_crs(epsg=4326)
    grid_wgs84 = grid_wgs84[grid_wgs84['count'] > 0]

    grid_wgs84.to_file(output_geojson, driver='GeoJSON')
    print(f"✅ 聚合完成！{len(grid_wgs84)} 個非空網格")
    print(f"📁 儲存至：{output_geojson}")

if __name__ == "__main__":
    # ✅ 已更新為你實際的資料夾名稱
    inlets_folder = "data/Drain_Gully_Grating_SHP"
    
    try:
        shp_path = find_shp(inlets_folder)
        aggregate_inlets(shp_path, "data/inlets_density.geojson", grid_size=500)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("請確認已將『進水渠蓋』SHP 檔案放入 data/Drain_Gully_Grating_SHP 資料夾")