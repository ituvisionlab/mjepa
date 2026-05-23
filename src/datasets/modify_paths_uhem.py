"""

import pandas as pd

in_csv  = "nc_ad_train_k8.csv.csv"
out_csv = "nc_ad_train_k8.csv_new.csv"

old = "/gpfs/data/sodicksonlab/gozde/"
new = "/ari/users/eergun01/data/"

df = pd.read_csv(in_csv)
df["nii_file_path"] = df["nii_file_path"].astype(str).str.replace(old, new, regex=False)
df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)

"""

import pandas as pd
import os

in_csv  = "nc_mci_train_k8.csv"
out_csv = "nc_mci_train_k8_new.csv"

old_prefix = "/gpfs/data/sodicksonlab/gozde/"
new_prefix = "/ari/users/eergun01/data/"

# -----------------------
# rewrite paths
# -----------------------
df = pd.read_csv(in_csv)

df["nii_file_path"] = (
    df["nii_file_path"]
    .astype(str)
    .str.replace(old_prefix, new_prefix, regex=False)
)

df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)

# -----------------------
# rename files
# -----------------------
base, ext = os.path.splitext(in_csv)
old_csv = base + "_old" + ext

# 1) move current -> _old
if not os.path.exists(old_csv):
    os.replace(in_csv, old_csv)
    print("Renamed old file to:", old_csv)
else:
    raise RuntimeError(f"{old_csv} already exists. Aborting to avoid overwrite.")

# 2) move _new -> normal name
os.replace(out_csv, in_csv)
print("Renamed new file to:", in_csv)
