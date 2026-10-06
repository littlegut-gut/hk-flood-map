# check_shp.py
import shapefile
import glob
import os

def check_shp_folder(folder_path):
    # 搵出 .shp 檔案
    shp_files = glob.glob(os.path.join(folder_path, "*.shp"))
    if not shp_files:
        print(f"❌ 在 {folder_path} 找不到 .shp 檔案")
        return
    
    shp_path = shp_files[0]
    print(f"📁 讀取：{shp_path}")
    
    try:
        sf = shapefile.Reader(shp_path, encoding='utf-8')
    except:
        sf = shapefile.Reader(shp_path, encoding='big5')
    
    # 顯示基本資訊
    print(f"📊 記錄數量：{len(sf.shapeRecords())}")
    print(f"📋 欄位名稱：{[f[0] for f in sf.fields[1:]]}")
    
    # 檢查頭 5 個記錄嘅幾何類型
    for i, rec in enumerate(sf.shapeRecords()[:5]):
        if rec.shape is None:
            print(f"  ⚠️ 記錄 {i+1}: 幾何為 None")
        else:
            print(f"  ✅ 記錄 {i+1}: 幾何類型 = {rec.shape.shapeType} ({rec.shape.__geo_interface__['type'] if hasattr(rec.shape, '__geo_interface__') else 'unknown'})")
    
    # 統計唔同幾何類型數量
    type_counts = {}
    for rec in sf.shapeRecords():
        if rec.shape is None:
            type_counts['NULL'] = type_counts.get('NULL', 0) + 1
        else:
            t = rec.shape.shapeType
            type_counts[t] = type_counts.get(t, 0) + 1
    
    print(f"\n📈 幾何類型統計：{type_counts}")

# 檢查沙井資料夾
print("===== 檢查沙井 SHP =====")
check_shp_folder("data/Manhole_Stormwater_SHP")

print("\n===== 檢查渠管 SHP =====")
check_shp_folder("data/Pipe_Stormwater_SHP")