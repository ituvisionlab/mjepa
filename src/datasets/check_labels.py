import pandas as pd

df = pd.read_csv("/gpfs/home/unalg01/jepa/src/datasets/adni_all_bet_train.csv")  # Adjust to your dataset file
print("Unique Labels:", df["label"].unique())