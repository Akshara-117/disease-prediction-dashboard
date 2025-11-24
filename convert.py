import pandas as pd
import os, glob

input_folder = r"C:\Users\aksha\OneDrive\Desktop\big data\trial"

# detect all excel files
excel_paths = glob.glob(os.path.join(input_folder, "*.xls")) + \
              glob.glob(os.path.join(input_folder, "*.xlsx"))

for path in excel_paths:
    try:
        df = pd.read_excel(path, engine="openpyxl") if path.endswith("xlsx") else pd.read_excel(path)
        out = os.path.splitext(path)[0] + ".csv"
        df.to_csv(out, index=False)
        print("Converted:", out)
    except Exception as e:
        print("Failed:", path, "->", e)
