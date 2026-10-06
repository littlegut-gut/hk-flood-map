# convert_drainage.py (修正版 - 跳過 NULL 幾何)
import os
import json
import glob
import shapefile
from pyproj import Transformer
from shapely.geometry import shape

def find_shp_file(folder_path):
    """在指定資料夾中尋找第一個 .shp 檔案"""
    shp_files = glob.glob(os.path.join(folder_path, "*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"在 {folder_path} 中找不到任何 .shp 檔案")
    return shp_files[0]

def convert_shp_to_geojson(input_shp, output_geojson, layer_name=""):
    """
    將 SHP 轉為 WGS84 GeoJSON，跳過 NULL 幾何，保留前三個屬性欄位
    """
    # 嘗試讀取（自動偵測編碼）
    try:
        sf = shapefile.Reader(input_shp, encoding='utf-8')
    except:
        sf = shapefile.Reader(input_shp, encoding='big5')

    # 取得欄位名稱（跳過第一個 DeletionFlag）
    field_names = [field[0] for field in sf.fields[1:]]
    print(f"📋 {layer_name} 的欄位：{field_names}")

    # 座標轉換器：HK1980 Grid (EPSG:2326) → WGS84 (EPSG:4326)
    transformer = Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)

    features = []
    skipped = 0
    for shape_rec in sf.shapeRecords():
        # ----- 跳過 NULL 幾何 (shapeType == 0) 或無效形狀 -----
        if shape_rec.shape is None or shape_rec.shape.shapeType == 0:
            skipped += 1
            continue

        try:
            geom = shape(shape_rec.shape.__geo_interface__)
        except Exception as e:
            print(f"⚠️ 跳過一個無法解析的幾何：{e}")
            skipped += 1
            continue

        # 根據幾何類型轉換座標
        if geom.geom_type == 'Point':
            x, y = transformer.transform(geom.x, geom.y)
            new_geom = {'type': 'Point', 'coordinates': [x, y]}
        elif geom.geom_type == 'LineString':
            new_coords = [transformer.transform(x, y) for x, y in geom.coords]
            new_geom = {'type': 'LineString', 'coordinates': new_coords}
        elif geom.geom_type == 'Polygon':
            exterior = [transformer.transform(x, y) for x, y in geom.exterior.coords]
            interiors = [[transformer.transform(x, y) for x, y in ring.coords] for ring in geom.interiors]
            new_geom = {'type': 'Polygon', 'coordinates': [exterior] + interiors}
        elif geom.geom_type == 'MultiLineString':
            new_lines = [[transformer.transform(x, y) for x, y in line.coords] for line in geom.geoms]
            new_geom = {'type': 'MultiLineString', 'coordinates': new_lines}
        else:
            # 其他類型（如 MultiPoint）暫時跳過
            skipped += 1
            continue

        # 提取屬性（保留頭三個欄位，可根據需要調整）
        attrs = shape_rec.record
        props = {}
        for i, name in enumerate(field_names):
            if i < 3:
                props[name] = attrs[i] if i < len(attrs) else None

        features.append({
            'type': 'Feature',
            'geometry': new_geom,
            'properties': props
        })

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    with open(output_geojson, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"✅ 轉換完成：{len(features)} 個有效要素，跳過 {skipped} 個無效記錄")
    print(f"📁 儲存至：{output_geojson}\n")

# ------------------ 執行 ------------------
if __name__ == "__main__":
    # 定義輸入資料夾和輸出檔案
    pipe_folder = "data/Pipe_Stormwater_SHP"
    manhole_folder = "data/Manhole_Stormwater_SHP"

    # 尋找 SHP 檔案
    pipe_shp = find_shp_file(pipe_folder)
    manhole_shp = find_shp_file(manhole_folder)

    # 轉換渠管
    convert_shp_to_geojson(pipe_shp, "data/pipe_rainwater.geojson", "渠管(雨水)")
    # 轉換沙井
    convert_shp_to_geojson(manhole_shp, "data/manhole_rainwater.geojson", "沙井(雨水)")

    print("🎉 所有轉換完成！")