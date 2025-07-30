import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from river.drift import ADWIN
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


CSV_FILE = "data/cleaned/issues_3.csv" 
INITIAL_TRAIN_SIZE = 100              
DELTA = 0.01                           

def prepare_data_from_df(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = [f"{t} {d}" for t, d in zip(df["Title"], df["Description"])]
    embeddings = model.encode(sentences)

    type_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    priority_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    type_encoded = type_ohe.fit_transform(df[["Type"]])
    priority_encoded = priority_ohe.fit_transform(df[["Priority"]])

    X = np.hstack((embeddings, type_encoded, priority_encoded))
    y = np.log1p(df["Result"].astype(float))

    return X, y, type_ohe, priority_ohe

df = pd.read_csv(CSV_FILE, on_bad_lines="skip")

X_all, y_all, type_ohe, priority_ohe = prepare_data_from_df(df)

model = RandomForestRegressor(n_estimators=100, random_state=42)
adwin = ADWIN(delta=DELTA)
errors = []
drift_points = []

X_train, y_train = X_all[:INITIAL_TRAIN_SIZE], y_all[:INITIAL_TRAIN_SIZE]
model.fit(X_train, y_train)

for i in range(INITIAL_TRAIN_SIZE, len(X_all)):
    x_i = X_all[i].reshape(1, -1)
    y_true = y_all[i]

    y_pred = model.predict(x_i)[0]
    error = abs(np.expm1(y_true) - np.expm1(y_pred)) / np.expm1(y_true)
    errors.append(error)

    adwin.update(error) 


    if adwin.drift_detected:
        print(f"Drift detected at index {i}")
        drift_points.append(i - INITIAL_TRAIN_SIZE)

        train_start = max(0, i - INITIAL_TRAIN_SIZE)
        model.fit(X_all[train_start:i], y_all[train_start:i])


plt.figure(figsize=(14, 6))
plt.plot(errors, label="Relative Error (MMRE)", linewidth=1)
for idx, dp in enumerate(drift_points):
    plt.axvline(x=dp, color='red', linestyle='--', label='Drift Detected' if idx == 0 else "")
plt.xlabel("Sample Index (relative to test start)")
plt.ylabel("Relative Error")
plt.title("ADWIN Concept Drift Detection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
