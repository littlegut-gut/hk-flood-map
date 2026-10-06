# verify_landuse.py
"""
土地利用因子驗證腳本
1. 相關性分析（runoff_coeff vs risk_score）
2. 分箱統計（各 C 值區間的平均風險）
3. 已知地點採樣（市區應高、郊野應低）
4. 歷史黑點驗證（黑點 300m 內平均風險）
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

RISK_PATH = "data/risk_zones.geojson"

print("=" * 60)
print("土地利用因子驗證")
print("=" * 60)

g = gpd.read_file(RISK_PATH)
print(f"載入 {len(g)} 個網格，CRS: {g.crs}\n")

# ============================================================
# 驗證 1a：相關性分析
# ============================================================
print("【驗證 1a】相關性分析")
print("-" * 60)

r = g['runoff_coeff'].corr(g['risk_score'])
print(f"  runoff_coeff  vs  risk_score    相關性: {r:+.3f}")

# 對比其他因子，看土地利用是否被其他因子蓋過
for col in ['elevation', 'inlet_count', 'dist_to_pipe',
            'population_density', 'dist_to_flood', 'coastal_factor']:
    if col in g.columns:
        rr = g[col].corr(g['risk_score'])
        print(f"  {col:20s} vs  risk_score    相關性: {rr:+.3f}")

print()
if r > 0.3:
    print("  ✅ 相關性 > 0.3，土地利用因子有效")
elif r > 0.15:
    print("  ⚠️ 相關性 0.15~0.3，土地利用有影響但被其他因子稀釋")
else:
    print("  ❌ 相關性 < 0.15，土地利用因子可能失效，需檢查映射")

# ============================================================
# 驗證 1b：分箱統計
# ============================================================
print("\n【驗證 1b】各 C 值區間的平均風險")
print("-" * 60)

bins   = [0, 0.3, 0.5, 0.7, 0.9, 1.01]
labels = ['低透水(<0.3)', '中低(0.3~0.5)', '中(0.5~0.7)',
          '中高(0.7~0.9)', '高(>0.9)']
g['_c_bin'] = pd.cut(g['runoff_coeff'], bins=bins, labels=labels)

summary = g.groupby('_c_bin', observed=True).agg(
    網格數=('risk_score', 'count'),
    平均風險=('risk_score', 'mean'),
    中位風險=('risk_score', 'median'),
    平均C值=('runoff_coeff', 'mean'),
).round(2)
print(summary.to_string())

# 檢查單調性
means = summary['平均風險'].values
if len(means) >= 2 and all(means[i] <= means[i+1] for i in range(len(means)-1)):
    print("\n  ✅ 平均風險隨 C 值單調遞增，因子方向正確")
else:
    print("\n  ⚠️ 平均風險非單調遞增，可能有交互作用或權重問題")

# ============================================================
# 驗證 1c：高/低 C 值區域對比
# ============================================================
print("\n【驗證 1c】極端組對比")
print("-" * 60)

high_c = g[g['runoff_coeff'] > 0.7]['risk_score']
low_c  = g[g['runoff_coeff'] < 0.25]['risk_score']
print(f"  高 C 值 (>0.7) ：{len(high_c):5d} 個網格，平均風險 {high_c.mean():.1f}")
print(f"  低 C 值 (<0.25)：{len(low_c):5d} 個網格，平均風險 {low_c.mean():.1f}")
print(f"  全港平均：{g['risk_score'].mean():.1f}")
diff = high_c.mean() - low_c.mean()
print(f"\n  高低差：{diff:+.1f} 分")
if diff > 5:
    print("  ✅ 高低 C 值區域風險差 > 5 分，因子有區分度")
elif diff > 0:
    print("  ⚠️ 差異存在但偏小（0~5 分）")
else:
    print("  ❌ 方向反了！高 C 值區域風險反而更低")

# ============================================================
# 驗證 2：已知地點採樣
# ============================================================
print("\n【驗證 2】已知地點採樣（WGS84 座標附近最近網格）")
print("-" * 60)

KNOWN_PLACES = {
    # 市區（保持不變）
    "旺角":         (22.3193, 114.1694, "高"),
    "深水埗":       (22.3302, 114.1622, "高"),
    "中環":         (22.2819, 114.1583, "高"),
    "觀塘":         (22.3165, 114.2155, "高"),
    "葵涌貨櫃碼頭": (22.3380, 114.1300, "高"),
    "赤鱲角機場":   (22.3087, 113.9145, "高"),
    "沙田市中心":   (22.3833, 114.1883, "中"),

    # 郊野/保護區（改用精確座標）
    "大帽山頂":     (22.4106, 114.1241, "低"),  # 山頂雷達站
    "米埔保護區":   (22.4922, 114.0356, "低"),  # 保護區中心
    "西貢東郊野":   (22.4060, 114.3380, "低"),  # 大浪灣一帶
    "船灣淡水湖":   (22.4680, 114.2300, "低"),  # 湖心
}

pts = gpd.GeoDataFrame(
    [{'name': k, 'expect': v[2],
      'geometry': Point(v[1], v[0])}
     for k, v in KNOWN_PLACES.items()],
    crs='EPSG:4326'
).to_crs(epsg=2326)

# ⚠️ 關鍵：把 g 投影到 2326 再匹配
g_utm = g.to_crs(epsg=2326)

matched = gpd.sjoin_nearest(
    pts,
    g_utm[['geometry', 'runoff_coeff', 'risk_score',
           'elevation', 'population_density', 'inlet_count']],
    how='left'
).drop_duplicates(subset='name')

print(f"  {'地點':<14s} {'期望':<4s} {'C值':>6s} {'風險':>6s} "
      f"{'海拔':>6s} {'人口':>8s} {'渠蓋':>5s}  判讀")
print("  " + "-" * 70)

correct = 0
total = 0
for _, row in matched.iterrows():
    name    = row['name']
    expect  = row['expect']
    c       = row['runoff_coeff']
    score   = row['risk_score']
    elev    = row['elevation']
    pop     = row['population_density']
    inlet   = int(row['inlet_count']) if pd.notna(row['inlet_count']) else 0

    if expect == "高":
        ok = score > 50
    elif expect == "低":
        ok = score < 45
    else:
        ok = 40 <= score <= 60

    mark = "✅" if ok else "⚠️"
    correct += int(ok)
    total += 1

    print(f"  {name:<14s} {expect:<4s} {c:>6.2f} {score:>6.1f} "
          f"{elev:>6.1f} {pop:>8.0f} {inlet:>5d}  {mark}")

print(f"\n  命中率：{correct}/{total} = {correct/total*100:.0f}%")
if correct / total >= 0.8:
    print("  ✅ 已知地點命中率 ≥ 80%，模型行為合理")
elif correct / total >= 0.6:
    print("  ⚠️ 命中率 60~80%，部分地點需檢討")
else:
    print("  ❌ 命中率 < 60%，模型可能有系統性問題")

# ============================================================
# 驗證 2b：歷史黑點驗證
# ============================================================
print("\n【驗證 2b】歷史積水黑點驗證")
print("-" * 60)

try:
    flood = gpd.read_file("data/FloodingBlackspots.geojson")
    if flood.crs is None:
        flood = flood.set_crs(epsg=4326, allow_override=True)
    flood = flood.to_crs(epsg=2326)

    blackspot_scores = []
    for geom in flood.geometry:
        buf = geom.buffer(300)
        mask = g_utm.geometry.intersects(buf)
        if mask.sum() > 0:
            blackspot_scores.append(g_utm.loc[mask, 'risk_score'].mean())

    if blackspot_scores:
        bs_mean = np.mean(blackspot_scores)
        hk_mean = g['risk_score'].mean()
        print(f"  黑點 300m 內平均風險：{bs_mean:.1f}")
        print(f"  全港平均：            {hk_mean:.1f}")
        print(f"  差異：                {bs_mean - hk_mean:+.1f} 分")
        if bs_mean > hk_mean + 8:
            print("  ✅ 黑點風險顯著高於全港，歷史因子有效")
        elif bs_mean > hk_mean:
            print("  ⚠️ 黑點略高但差距 < 8 分")
        else:
            print("  ❌ 黑點風險未高於全港，歷史因子可能失效")
    else:
        print("  ⚠️ 4 個黑點 300m 內均無網格")
except Exception as e:
    print(f"  ⚠️ 黑點驗證失敗：{e}")

# ============================================================
# 總結
# ============================================================
print("\n" + "=" * 60)
print("驗證完成。若上述指標都通過，可執行 risk_map.py 看地圖。")
print("=" * 60)