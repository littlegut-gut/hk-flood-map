# check_inlets.py
import json
import os

file_path = "data/inlets_density.geojson"

if not os.path.exists(file_path):
    print(f"❌ 檔案不存在：{file_path}")
else:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    count = len(data.get('features', []))
    print(f"✅ 進水渠蓋網格數量：{count}")