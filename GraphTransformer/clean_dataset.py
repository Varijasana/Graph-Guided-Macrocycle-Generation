import os
import pandas as pd

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = "/content/Macro-EquiDiff/MacTransformer/datasets/data"

# THIS is your RAW dataset (must exist)
INPUT_FILE = "data.csv"     # or your real raw file name

# THIS will be created
OUTPUT_FILE = "train.csv"

input_path = os.path.join(BASE_DIR, INPUT_FILE)
output_path = os.path.join(BASE_DIR, OUTPUT_FILE)

print("Reading raw file:", input_path)

# ----------------------------
# LOAD RAW DATA
# ----------------------------
df = pd.read_csv(input_path)

print("Original size:", len(df))
print("Columns:", df.columns)

# ----------------------------
# CREATE TRAIN DATA
# ----------------------------
if "src" in df.columns and "tgt" in df.columns:
    data = df[["src", "tgt"]]

elif "fragments" in df.columns:
    data = pd.DataFrame({
        "src": df["fragments"].astype(str),
        "tgt": df["fragments"].astype(str)
    })

else:
    raise ValueError("Need columns: src/tgt or fragments")

# ----------------------------
# CLEAN
# ----------------------------
data = data.dropna()
data = data[data["src"].str.len() > 0]
data = data[data["tgt"].str.len() > 0]

# ----------------------------
# SAVE train.csv
# ----------------------------
data.to_csv(output_path, index=False)

print("Saved train dataset at:", output_path)
print("Final size:", len(data))