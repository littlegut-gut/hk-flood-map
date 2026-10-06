# accumulate_rainfall.py
"""累積 HKO 每小時雨量到本地 JSON（按小時去重）。"""
import os
import json
import time
import requests
from datetime import datetime, timedelta

HISTORY_PATH = "data/rainfall_history.json"
HKO_URL = ("https://data.weather.gov.hk/weatherAPI/opendata/"
           "hourlyRainfall.php?lang=zh&station=all")
MAX_AGE_HOURS = 168

HKO_HEADERS = {
    'User-Agent': 'HK-Flood-Risk-Map/1.0 (research; contact: hk-flood@example.com)',
    'Accept': 'application/json',
}

def fetch_rainfall(max_retries=3):
    """從 HKO 取得當前雨量（含重試）。"""
    for attempt in range(max_retries):
        try:
            r = requests.get(HKO_URL, headers=HKO_HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   重試 {attempt+1}/{max_retries}...")
                time.sleep(5)
                continue
            raise
    obs_time = data.get('obsTime', '')
    rainfall = {}
    for item in data.get('hourlyRainfall', []):
        sid = item.get('automaticWeatherStationID')
        val = item.get('value')
        if sid and val is not None:
            try:
                rainfall[sid] = float(val)
            except (ValueError, TypeError):
                rainfall[sid] = 0.0
    return rainfall, obs_time

def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=1)


def hour_key(iso_time):
    """'2026-10-02T16:23:00+08:00' → '2026-10-02T16'（精確到小時）。"""
    if not iso_time or len(iso_time) < 13:
        return iso_time
    return iso_time[:13]


def prune_old(history, max_hours=MAX_AGE_HOURS):
    """移除超過 max_hours 的舊記錄。"""
    if not history:
        return history
    cutoff = datetime.now() - timedelta(hours=max_hours)
    pruned = []
    for entry in history:
        try:
            t = datetime.fromisoformat(entry['time'].replace('+08:00', ''))
            if t >= cutoff:
                pruned.append(entry)
        except Exception:
            pruned.append(entry)
    return pruned


def main():
    import sys
    log_path = "data/accumulate.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # 同時輸出到 console 和 log
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    log_file = open(log_path, 'a', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_file)
    
    from datetime import datetime
    print(f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print("🌧️ 累積雨量歷史（按小時去重）...")
    history = load_history()
    print(f"   現有記錄：{len(history)} 筆")

    try:
        rainfall, obs_time = fetch_rainfall()
    except Exception as e:
        print(f"   ❌ 獲取失敗：{e}")
        return

    if not obs_time:
        print("   ❌ API 未返回 obsTime")
        return

    # ---- 按小時去重 ----
    new_key = hour_key(obs_time)
    idx = None
    for i, entry in enumerate(history):
        if hour_key(entry['time']) == new_key:
            idx = i
            break

    n_rain = sum(1 for v in rainfall.values() if v > 0)

    if idx is not None:
        # 同一小時已有記錄 → 更新（保留 obsTime 較新的一筆）
        old_time = history[idx]['time']
        # 比較字串長度即可（ISO 格式按時間順序）
        if obs_time > old_time:
            history[idx] = {'time': obs_time, 'rainfall': rainfall}
            print(f"   🔄 更新 {new_key}（{old_time[11:16]} → {obs_time[11:16]}）"
                  f"，{len(rainfall)} 站（{n_rain} 站有雨）")
        else:
            print(f"   ℹ️ {new_key} 已有較新記錄（{old_time[11:16]}），跳過")
    else:
        # 新小時 → 新增一筆
        history.append({'time': obs_time, 'rainfall': rainfall})
        print(f"   ✅ 新增 {obs_time}，{len(rainfall)} 站（{n_rain} 站有雨）")

    history = prune_old(history)
    history.sort(key=lambda x: x['time'])
    save_history(history)
    print(f"   💾 儲存 {len(history)} 筆記錄 → {HISTORY_PATH}")


if __name__ == "__main__":
    main()