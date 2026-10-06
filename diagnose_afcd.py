# diagnose_afcd.py
import geopandas as gpd
import numpy as np

AFCD = "data/afcd_country_parks.geojson"

afcd = gpd.read_file(AFCD)
print(f"共 {len(afcd)} 個 feature\n")

# 只保留 polygon
poly = afcd[afcd.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
poly_utm = poly.to_crs(epsg=2326)
poly['area_ha'] = poly_utm.geometry.area.values / 10000

# 質心經緯度
poly_wgs = poly.to_crs(epsg=4326)
centroids = poly_wgs.geometry.centroid
poly['lon'] = centroids.x.round(5)
poly['lat'] = centroids.y.round(5)

# 名稱列
name_col = None
for c in ['name:zh', 'name', 'name:en']:
    if c in poly.columns:
        name_col = c
        break
poly['_name'] = poly[name_col].astype(str)

# ============================================================
# 1. 按面積排序（前 30）
# ============================================================
print("=" * 90)
print("【按面積排序（前 30）】")
print("=" * 90)
df = poly[['_name', 'area_ha', 'lon', 'lat', 'geometry']].copy()
print(df.nlargest(30, 'area_ha')[['_name', 'area_ha', 'lon', 'lat']].to_string())

# ============================================================
# 2. 檢查質心落在市區 3km 內
# ============================================================
CITY_POINTS = {
    "旺角":   (22.3193, 114.1694),
    "深水埗": (22.3302, 114.1622),
    "中環":   (22.2819, 114.1583),
    "觀塘":   (22.3165, 114.2155),
    "尖沙咀": (22.2975, 114.1722),
    "沙田":   (22.3833, 114.1883),
    "將軍澳": (22.3075, 114.2600),
    "荃灣":   (22.3710, 114.1100),
    "屯門":   (22.3910, 113.9770),
    "天水圍": (22.4600, 114.0000),
    "葵涌":   (22.3600, 114.1300),
    "啟德":   (22.3060, 114.2130),
    "西貢市": (22.3820, 114.2700),
}

print("\n" + "=" * 90)
print("【質心落在市區 3km 內的 feature】")
print("=" * 90)
n_suspect = 0
for _, row in poly.iterrows():
    for city, (clat, clon) in CITY_POINTS.items():
        dlat = abs(row['lat'] - clat) * 111
        dlon = abs(row['lon'] - clon) * 111 * np.cos(np.radians(clat))
        dist_km = np.sqrt(dlat**2 + dlon**2)
        if dist_km < 3:
            n_suspect += 1
            print(f"  {row['_name']:<40s} 面積 {row['area_ha']:>8.1f} ha "
                  f"質心距{city} {dist_km:.2f} km")
            break
if n_suspect == 0:
    print("  （無）")

# ============================================================
# 3. 面積 > 3000 公頃
# ============================================================
print("\n" + "=" * 90)
print("【面積 > 3000 公頃（香港最大郊野公園約 5600 公頃）】")
print("=" * 90)
big = df[df['area_ha'] > 3000].sort_values('area_ha', ascending=False)
print(big[['_name', 'area_ha', 'lon', 'lat']].to_string())

# ============================================================
# 4. 名稱含「海岸」或「Marine」
# ============================================================
print("\n" + "=" * 90)
print("【名稱含『海岸』或『Marine』的 feature】")
print("=" * 90)
marine = poly[poly['_name'].str.contains('海岸|Marine', case=False, na=False)]
print(marine[['_name', 'area_ha', 'lon', 'lat']].to_string())

# ============================================================
# 5. 面積 < 10 公頃（碎片）
# ============================================================
print("\n" + "=" * 90)
print(f"【面積 < 10 公頃（碎片）】共 {len(df[df['area_ha'] < 10])} 個")
print("=" * 90)

# ============================================================
# 6. 與市區網格重疊面積（用現有 risk_zones.geojson 比對）
# ============================================================
print("\n" + "=" * 90)
print("【每個 feature 內的高人口網格數（>500 人）】")
print("=" * 90)

try:
    risk = gpd.read_file("data/risk_zones.geojson").to_crs(epsg=2326)
    high_pop = risk[risk['population_total'] > 500]

    print(f"  高人口網格共 {len(high_pop)} 個")
    print(f"  {'名稱':<40s} {'面積ha':>8s} {'高人口網格數':>12s}")
    print("  " + "-" * 70)

    for _, row in poly_utm.iterrows():
        intersects = high_pop.geometry.intersects(row['geometry'])
        n = intersects.sum()
        if n > 0:
            name = poly.loc[row.name, '_name']
            area = row['area_ha'] if '_name' in poly.columns else 0
            print(f"  {name:<40s} {area:>8.1f} {n:>12d}")
except Exception as e:
    print(f"  跳過：{e}")