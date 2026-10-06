# check_afcd.py
import geopandas as gpd
import json

PATH = "data/afcd_country_parks.geojson"

# 1. 原始 JSON 檢查
with open(PATH, encoding='utf-8') as f:
    raw = json.load(f)

print(f"GeoJSON type: {raw.get('type')}")
print(f"Feature 數量: {len(raw.get('features', []))}")

geom_types = {}
for feat in raw.get('features', []):
    gt = feat.get('geometry', {}).get('type', 'None')
    geom_types[gt] = geom_types.get(gt, 0) + 1
print(f"幾何類型分佈: {geom_types}")

# 2. 用 GeoPandas 讀取
gdf = gpd.read_file(PATH)
print(f"\nGeoPandas 讀取: {len(gdf)} 個 feature")
print(f"CRS: {gdf.crs}")
print(f"幾何類型: {gdf.geometry.geom_type.value_counts().to_dict()}")
print(f"欄位: {list(gdf.columns)}")

# 3. 列出前 30 個名稱
print("\n前 30 個邊界名稱:")
for i, row in gdf.head(30).iterrows():
    name = (row.get('name') or row.get('name:zh')
            or row.get('name:en') or row.get('name_en') or '?')
    print(f"  {i+1:2d}. {name}")

# 4. 邊界檢查：投影後看覆蓋範圍
gdf_utm = gdf.to_crs(epsg=2326)
print(f"\nUTM 範圍: {gdf_utm.total_bounds}")
print(f"預期香港範圍約: [799000, 799000, 864000, 848000]")