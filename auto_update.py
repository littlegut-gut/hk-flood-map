# auto_update.py
import schedule
import time
from rainfall_map import create_rainfall_map

def job():
    print("🔄 開始更新雨量地圖...")
    create_rainfall_map()
    print("✅ 更新完成")

# 立即執行一次
job()

# 每 15 分鐘執行一次
schedule.every(15).minutes.do(job)

print("⏰ 排程啟動，每 15 分鐘更新一次")
while True:
    schedule.run_pending()
    time.sleep(30)