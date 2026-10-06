# test_overpass.py
import requests, json, time

HEADERS = {
    'User-Agent': 'HK-Flood-Risk-Map/1.0 (research; contact: hk-flood@example.com)',
    'Accept': 'application/json',
    'Content-Type': 'application/x-www-form-urlencoded',
}
HK_BBOX = "22.13,113.82,22.58,114.45"

query = f"""
[out:json][timeout:120];
(
  rel[boundary=national_park]({HK_BBOX});
  rel[boundary=protected_area]({HK_BBOX});
  rel[leisure=nature_reserve]({HK_BBOX});
);
out tags;
"""

for ep in [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]:
    print(f"\n→ {ep}")
    try:
        r = requests.post(ep, data={'data': query}, headers=HEADERS, timeout=120)
        print(f"  HTTP {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  ✅ {len(data['elements'])} 個 relation")
            for e in data['elements'][:30]:
                t = e.get('tags', {})
                print(f"     {t.get('name', t.get('name:en', '?')):35s} "
                      f"| {t.get('boundary', t.get('leisure', '?'))} "
                      f"| operator={t.get('operator', '-')}")
            break
        else:
            print(f"  {r.text[:200]}")
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")
    time.sleep(3)