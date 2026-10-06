# check_inpark.py
"""
郊野公園標記驗證
1. 統計 in_park 的整體分佈
2. 檢查公園內是否有異常人口網格
3. 對照「應在公園內/外」的已知地點
4. 特別檢查海岸公園是否誤標陸地網格
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from pyproj import Transformer

RISK_PATH = "data/risk_zones.geojson"

print("=" * 60)
print("郊野公園標記驗證")
print("=" * 60)

g = gpd.read_file(RISK_PATH)
g_utm = g.to_crs(epsg=2326)
print(f"載入 {len(g)} 個網格\n")

# ============================================================
# 檢查 1：整體分佈
# ============================================================
print("【檢查 1】in_park 整體分佈")
print("-" * 60)

if 'in_park' not in g.columns:
    print("  ❌ 找不到 in_park 欄位！請先跑 risk_calculator.py")
    raise SystemExit(1)

n_park = g['in_park'].sum()
n_total = len(g)
print(f"  公園內：{n_park:6d} 個（{n_park/n_total*100:.1f}%）")
print(f"  公園外：{n_total-n_park:6d} 個（{(n_total-n_park)/n_total*100:.1f}%）")

# 香港郊野公園 + 特別地區陸地面積約佔 40~45%
if 0.35 <= n_park/n_total <= 0.50:
    print("  ✅ 佔比在合理範圍（35~50%）")
elif n_park/n_total > 0.55:
    print("  ⚠️ 佔比過高，可能誤標了海岸公園或市區")
else:
    print("  ⚠️ 佔比偏低，可能漏標")

# ============================================================
# 檢查 2：公園內的人口分佈
# ============================================================
print("\n【檢查 2】公園內的人口分佈")
print("-" * 60)

park = g[g['in_park'] == True]
non_park = g[g['in_park'] == False]

pop_stats = park['population_total'].describe().round(1)
print(f"  公園內人口統計：")
print(f"    mean={pop_stats['mean']:.1f}, median={pop_stats['50%']:.1f}, "
      f"max={pop_stats['max']:.0f}")

n_high_pop = (park['population_total'] > 100).sum()
n_very_high = (park['population_total'] > 500).sum()
print(f"  公園內人口 > 100 的網格：{n_high_pop} 個 "
      f"（{n_high_pop/len(park)*100:.1f}%）")
print(f"  公園內人口 > 500 的網格：{n_very_high} 個")

if n_high_pop / len(park) < 0.05:
    print("  ✅ 公園內高人口網格佔比 < 5%，邊界合理")
elif n_high_pop / len(park) < 0.10:
    print("  ⚠️ 公園內高人口網格佔比 5~10%，可能邊界略寬")
else:
    print("  ❌ 公園內高人口網格佔比 > 10%，邊界嚴重污染")

# ============================================================
# 檢查 3：公園內外的 C 值對比
# ============================================================
print("\n【檢查 3】公園內外的 C 值對比")
print("-" * 60)
print(f"  公園內 C 值：mean={park['runoff_coeff'].mean():.3f}, "
      f"max={park['runoff_coeff'].max():.3f}")
print(f"  公園外 C 值：mean={non_park['runoff_coeff'].mean():.3f}, "
      f"max={non_park['runoff_coeff'].max():.3f}")

if park['runoff_coeff'].max() <= 0.101:
    print("  ✅ 公園內 C 值全部 = 0.10（屏蔽生效）")
else:
    n_violate = (park['runoff_coeff'] > 0.101).sum()
    print(f"  ⚠️ 公園內有 {n_violate} 個網格 C 值 > 0.10")

# ============================================================
# 檢查 4：應在公園外的地點
# ============================================================
print("\n【檢查 4】應在公園外的地點（不應被標為 in_park）")
print("-" * 60)

EXPECT_OUT = {
    "葵涌貨櫃碼頭":  (22.3380, 114.1300),
    "屯門工業區":    (22.3930, 113.9750),
    "將軍澳工業邨":  (22.2880, 114.2700),
    "香港國際機場":  (22.3087, 113.9145),
    "啟德郵輪碼頭":  (22.3060, 114.2130),
    "觀塘市中心":    (22.3165, 114.2155),
    "旺角":          (22.3193, 114.1694),
    "中環":          (22.2819, 114.1583),
    "深水埗":        (22.3302, 114.1622),
    "沙田市中心":    (22.3833, 114.1883),
    "天水圍":        (22.4600, 114.0000),
}

transformer = Transformer.from_crs('EPSG:4326', 'EPSG:2326', always_xy=True)

print(f"  {'地點':<16s} {'in_park':>8s} {'C值':>6s} {'人口':>8s}  判讀")
print("  " + "-" * 60)

n_wrong = 0
wrong_names = []
for name, (lat, lng) in EXPECT_OUT.items():
    x, y = transformer.transform(lng, lat)
    dists = g_utm.geometry.centroid.distance(Point(x, y))
    row = g_utm.loc[dists.idxmin()]

    in_park = bool(row['in_park'])
    c = row['runoff_coeff']
    pop = row['population_total']

    ok = not in_park
    if not ok:
        n_wrong += 1
        wrong_names.append(name)
    mark = "✅" if ok else "❌ 誤標"
    print(f"  {name:<16s} {str(in_park):>8s} {c:>6.2f} {pop:>8.0f}  {mark}")

print(f"\n  誤標：{n_wrong}/{len(EXPECT_OUT)}")
if n_wrong == 0:
    print("  ✅ 所有「應在公園外」地點均正確")
else:
    print(f"  ❌ 誤標地點：{wrong_names}")
    print("  → 建議過濾海岸公園邊界（含『海岸公園』或『Marine Park』的 feature）")

# ============================================================
# 檢查 5：應在公園內的地點
# ============================================================
print("\n【檢查 5】應在公園內的地點（應被標為 in_park）")
print("-" * 60)

EXPECT_IN = {
    "大帽山頂":      (22.4106, 114.1241),
    "城門水塘":      (22.3860, 114.1450),
    "西貢東郊野":    (22.4060, 114.3380),
    "船灣郊野公園":  (22.4680, 114.2400),
    "大潭郊野公園":  (22.2600, 114.1900),
    "獅子山郊野":    (22.3550, 114.1750),
    "金山郊野公園":  (22.3600, 114.1400),
    "林村郊野公園":  (22.4500, 114.1000),
}

print(f"  {'地點':<16s} {'in_park':>8s} {'C值':>6s} {'海拔':>6s}  判讀")
print("  " + "-" * 60)

n_miss = 0
miss_names = []
for name, (lat, lng) in EXPECT_IN.items():
    x, y = transformer.transform(lng, lat)
    dists = g_utm.geometry.centroid.distance(Point(x, y))
    row = g_utm.loc[dists.idxmin()]

    in_park = bool(row['in_park'])
    c = row['runoff_coeff']
    elev = row['elevation']

    ok = in_park
    if not ok:
        n_miss += 1
        miss_names.append(name)
    mark = "✅" if ok else "❌ 漏標"
    print(f"  {name:<16s} {str(in_park):>8s} {c:>6.2f} {elev:>6.1f}  {mark}")

print(f"\n  漏標：{n_miss}/{len(EXPECT_IN)}")
if n_miss == 0:
    print("  ✅ 所有「應在公園內」地點均正確標記")
else:
    print(f"  ⚠️ 漏標地點：{miss_names}")
    print("  → 可能是採樣點剛好在邊界上，或該公園在 OSM 中邊界缺失")

# ============================================================
# 檢查 6：高人口且被標為 in_park 的網格（重點排查）
# ============================================================
print("\n【檢查 6】高人口（>500）但被標為 in_park 的網格")
print("-" * 60)

suspect = park[park['population_total'] > 500].copy()
print(f"  共 {len(suspect)} 個可疑網格")

if len(suspect) > 0:
    # 轉回 WGS84 顯示經緯度
    suspect_wgs = suspect.to_crs(epsg=4326)
    suspect_wgs['lon'] = suspect_wgs.geometry.centroid.x.round(4)
    suspect_wgs['lat'] = suspect_wgs.geometry.centroid.y.round(4)

    print("\n  前 20 個可疑網格（按人口排序）：")
    print(f"  {'lat':>9s} {'lon':>10s} {'人口':>8s} {'C值':>6s} {'海拔':>7s}")
    print("  " + "-" * 50)
    for _, r in suspect_wgs.nlargest(20, 'population_total').iterrows():
        print(f"  {r['lat']:>9.4f} {r['lon']:>10.4f} "
              f"{r['population_total']:>8.0f} "
              f"{r['runoff_coeff']:>6.2f} {r['elevation']:>7.1f}")

    if len(suspect) > 100:
        print(f"\n  ❌ {len(suspect)} 個高人口網格被誤標，建議過濾海岸公園")
    elif len(suspect) > 20:
        print(f"\n  ⚠️ {len(suspect)} 個可疑網格，可能邊界略寬")
    else:
        print(f"\n  ✅ 可疑網格數量少（< 20），可能是邊界 200m 誤差")

# ============================================================
# 總結
# ============================================================
print("\n" + "=" * 60)
print("檢查完成")
print("=" * 60)

issues = []
if n_wrong > 0:
    issues.append(f"應在公園外卻被標記：{n_wrong} 個")
if n_miss > 0:
    issues.append(f"應在公園內卻漏標：{n_miss} 個")
if n_high_pop / len(park) > 0.10:
    issues.append(f"公園內高人口網格佔比過高：{n_high_pop/len(park)*100:.1f}%")

if not issues:
    print("✅ 所有檢查通過，in_park 標記正確")
    print("下一步：執行 python risk_map.py 生成地圖")
else:
    print("⚠️ 發現以下問題：")
    for issue in issues:
        print(f"  - {issue}")
    print("\n建議：在 risk_calculator.py 過濾海岸公園邊界後重跑")