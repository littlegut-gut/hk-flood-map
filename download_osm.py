import requests
import json
import os

def download_osm_country_parks(output_path="data/afcd_country_parks.geojson"):
    """從 Overpass API 提取香港郊野公園邊界（多邊形）。"""
    if os.path.exists(output_path):
        print(f"   ✅ 已存在：{output_path}")
        return output_path

    overpass_url = "https://overpass-api.de/api/interpreter"
    query = """
    [out:json][timeout:120];
    area["name:en"="Hong Kong"]->.hk;
    (
      rel[boundary=national_park](area.hk);
      rel[boundary=protected_area]["operator"~"AFCD|漁農"](area.hk);
    );
    out geom;
    """
    try:
        r = requests.post(overpass_url, data={'data': query}, timeout=180)
        r.raise_for_status()
        data = r.json()

        features = []
        for elem in data.get('elements', []):
            if elem['type'] != 'relation':
                continue
            tags = elem.get('tags', {})
            name = tags.get('name', tags.get('name:en', 'Unknown'))

            # 收集所有 way 的座標，組裝為 MultiLineString
            rings = []
            for member in elem.get('members', []):
                if member['type'] == 'way' and 'geometry' in member:
                    coords = [[pt['lon'], pt['lat']]
                              for pt in member['geometry']]
                    if len(coords) >= 4:
                        rings.append(coords)

            if rings:
                features.append({
                    'type': 'Feature',
                    'properties': {
                        'name': name,
                        'name_en': tags.get('name:en', ''),
                        'operator': tags.get('operator', 'AFCD'),
                    },
                    'geometry': {
                        'type': 'MultiLineString',
                        'coordinates': rings,
                    }
                })

        geojson = {'type': 'FeatureCollection', 'features': features}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False)
        print(f"   ✅ 下載 {len(features)} 個郊野公園邊界：{output_path}")
        return output_path

    except Exception as e:
        print(f"   ⚠️ Overpass 下載失敗：{e}")
        return None