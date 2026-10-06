# diagnose_osm.py
import requests
import json

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# 香港 bounding box: south,west,north,east
HK_BBOX = "22.13,113.82,22.58,114.45"


def run_query(name, query):
    print(f"\n{'='*60}")
    print(f"【{name}】")
    print(f"{'='*60}")
    print(f"Query:\n{query.strip()}\n")

    for endpoint in OVERPASS_ENDPOINTS:
        print(f"→ 嘗試 {endpoint}")
        try:
            r = requests.post(endpoint, data={'data': query}, timeout=120)
            print(f"  HTTP {r.status_code}")
            if r.status_code != 200:
                print(f"  回應前 300 字：{r.text[:300]}")
                continue
            data = r.json()
            n = len(data.get('elements', []))
            print(f"  ✅ 返回 {n} 個 element")
            if n > 0:
                for e in data['elements'][:5]:
                    tags = e.get('tags', {})
                    name_val = tags.get('name', tags.get('name:en', '?'))
                    print(f"     type={e['type']}, id={e.get('id')}, name={name_val}")
            return data
        except Exception as e:
            print(f"  ❌ ERROR: {type(e).__name__}: {e}")

    print("  ⚠️ 所有端點均失敗")
    return None


# ---- 測試 1：用 bbox 查 boundary=national_park ----
run_query("bbox + boundary=national_park", f"""
[out:json][timeout:60];
rel[boundary=national_park]({HK_BBOX});
out tags;
""")

# ---- 測試 2：用 bbox 查 boundary=protected_area ----
run_query("bbox + boundary=protected_area", f"""
[out:json][timeout:60];
rel[boundary=protected_area]({HK_BBOX});
out tags;
""")

# ---- 測試 3：用 bbox 查 leisure=nature_reserve ----
run_query("bbox + leisure=nature_reserve", f"""
[out:json][timeout:60];
rel[leisure=nature_reserve]({HK_BBOX});
out tags;
""")

# ---- 測試 4：用 area 查（原本的寫法）----
run_query("area name:en=Hong Kong + boundary=national_park", """
[out:json][timeout:60];
area["name:en"="Hong Kong"]->.hk;
rel[boundary=national_park](area.hk);
out tags;
""")

# ---- 測試 5：用 area 查 protected_area ----
run_query("area name:en=Hong Kong + boundary=protected_area", """
[out:json][timeout:60];
area["name:en"="Hong Kong"]->.hk;
rel[boundary=protected_area](area.hk);
out tags;
""")