import os
folders = [
    'Multiple_Pipes_Stormwater',
    'Manhole_Stormwater_SHP',
    'Pipe_Stormwater_SHP',
    'Drain_Gully_Grating_SHP'
]
for folder in folders:
    path = f"data/{folder}"
    if os.path.exists(path):
        shp_files = [f for f in os.listdir(path) if f.endswith('.shp')]
        print(f"{folder}: {shp_files}")
    else:
        print(f"{folder}: ❌ 文件夹不存在")