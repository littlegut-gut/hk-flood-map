# check_history.py
import json

with open('data/rainfall_history.json', encoding='utf-8') as f:
    h = json.load(f)

print(f'共 {len(h)} 筆記錄\n')
for e in h:
    rain = e['rainfall']
    n_rain = sum(1 for v in rain.values() if v > 0)
    max_rain = max(rain.values()) if rain else 0
    max_sid = max(rain, key=rain.get) if rain else '--'
    print(f'  {e["time"]}  {n_rain:2d}站有雨  最大 {max_rain:5.1f}mm ({max_sid})')
    