# check_shp_geometry.py
import shapefile
import glob

shp_files = glob.glob("data/Drain_Gully_Grating_SHP/*.shp")
if not shp_files:
    print("❌ 找不到 .shp 文件")
else:
    sf = shapefile.Reader(shp_files[0])
    print(f"記錄總數：{len(sf.shapeRecords())}")
    non_null = sum(1 for rec in sf.shapeRecords() if rec.shape and rec.shape.shapeType != 0)
    print(f"包含幾何的記錄數：{non_null}")