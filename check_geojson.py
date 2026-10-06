import geopandas as gpd
flood = gpd.read_file("data/FloodingBlackspots.geojson")
print(f"记录数：{len(flood)}")
print(f"坐标系：{flood.crs}")
print(f"几何类型：{flood.geom_type.unique()}")
print(f"字段列表：{list(flood.columns)}")
print(f"\n前3行：")
print(flood.head(3))