# check_tc.py
import os
import glob

TC_DIR = "data/tc"

print(f"檢查目錄：{os.path.abspath(TC_DIR)}")
print(f"存在：{os.path.isdir(TC_DIR)}")
print()

if os.path.isdir(TC_DIR):
    files = os.listdir(TC_DIR)
    print(f"檔案數：{len(files)}")
    for f in files:
        fp = os.path.join(TC_DIR, f)
        size = os.path.getsize(fp)
        print(f"  {f}  ({size:,} bytes)")
    
    # 印出每個檔案的前 30 行
    for fp in glob.glob(os.path.join(TC_DIR, "hko_tctrack_*.*")):
        print(f"\n{'='*60}")
        print(f"檔案：{fp}")
        print('='*60)
        with open(fp, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= 30:
                    print("  ...")
                    break
                print(f"  {i+1:3d} | {line.rstrip()}")
else:
    print(f"❌ 目錄不存在：{TC_DIR}")
    print("   請先建立並放入 HKO 的 TC 路徑檔案")