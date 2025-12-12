import json, joblib
import pandas as pd

ART = "artifacts/rossmann_v1"

meta = json.load(open(f"{ART}/meta.json"))
sigma = json.load(open(f"{ART}/sigma.json"))
model = joblib.load(f"{ART}/lgbm_model.joblib")

train = pd.read_parquet("data/processed/rossmann/train_merged_clean.parquet")
test  = pd.read_parquet("data/processed/rossmann/test_merged.parquet")
sub   = pd.read_csv(f"{ART}/submission.csv")

print("Model loaded. #features:", len(meta["feature_cols"]))
print("Sigma global:", sigma["sigma_global"], "stores:", len(sigma["sigma_store"]))
print("Train/Test/Sub shapes:", train.shape, test.shape, sub.shape)
