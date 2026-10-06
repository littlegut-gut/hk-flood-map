# test_rainfall.py
import requests
import json

url = "https://data.weather.gov.hk/weatherAPI/opendata/hourlyRainfall.php?lang=zh&station=all"
r = requests.get(url)
data = r.json()

print("===== 原始 JSON 結構 (前 800 字元) =====")
print(json.dumps(data, indent=2, ensure_ascii=False)[:800])
print("\n===== 頂層鍵 =====")
print(data.keys())

# 自動尋找包含站點列表的鍵
stations = None
if 'stations' in data:
    stations = data['stations']
elif 'station' in data:
    stations = data['station']
elif 'data' in data and isinstance(data['data'], dict):
    for key in ['stations', 'station']:
        if key in data['data']:
            stations = data['data'][key]
            break

if stations is None:
    print("❌ 找不到站點列表，請檢查上述 JSON 結構。")
else:
    print(f"✅ 找到 {len(stations)} 個站點，例如：")
    for s in stations[:3]:
        print(s)