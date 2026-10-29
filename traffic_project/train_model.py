"""
train_model.py
--------------
Full ML pipeline: load -> clean -> leakage check -> feature engineering ->
train 3 models -> evaluate -> save model, metrics and charts.
Run:  python train_model.py
"""
import json, os, warnings
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

DATA = "data/traffic_data.csv"
CLASSES = ["Low", "Medium", "High", "Severe"]
# Columns that measure congestion itself (the target is computed from / mirrors them).
LEAKY = ["Traffic_Volume", "Average_Speed", "Travel_Time_Index", "Road_Capacity_Utilization"]
os.makedirs("models", exist_ok=True); os.makedirs("static/plots", exist_ok=True)

# ---------- 1. Load + clean ----------
df = pd.read_csv(DATA)
df.columns = [c.strip().replace("/", "_").replace(" ", "_") for c in df.columns]
ren = {"Road_Intersection_Name": "Road_Name", "Weather_Conditions": "Weather_Condition",
       "Congestion_Level": "Congestion_Level"}
df = df.rename(columns=ren)
print("Shape:", df.shape); print(df.dtypes)
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
before = len(df); df = df.drop_duplicates(); print("Duplicates removed:", before - len(df))
df = df.dropna(subset=["Congestion_Level"])
for c in df.columns:                       # fill missing: median for numbers, mode for text
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median() if df[c].dtype.kind in "if" else df[c].mode()[0])

# Target -> 4 classes (if the target is numeric, split into quartile-based classes)
if df["Congestion_Level"].dtype.kind in "if":
    df["Congestion_Level"] = pd.qcut(df["Congestion_Level"], 4, labels=CLASSES)
df["Congestion_Level"] = df["Congestion_Level"].astype(str).str.title()
df = df[df["Congestion_Level"].isin(CLASSES)].reset_index(drop=True)

# ---------- 2. Feature engineering ----------
if "Date" in df.columns:
    d = pd.to_datetime(df["Date"], errors="coerce")
    for col, val in [("Hour", d.dt.hour), ("Day", d.dt.day), ("Month", d.dt.month), ("DayOfWeek", d.dt.dayofweek)]:
        if col not in df.columns: df[col] = val
df["Is_Weekend"] = (df["DayOfWeek"] >= 5).astype(int)
if "Peak_Hour" not in df.columns:
    df["Peak_Hour"] = df["Hour"].isin([7, 8, 9, 10, 17, 18, 19, 20]).astype(int)
df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
df["Roadwork_Flag"] = df["Roadwork_and_Construction_Activity"].astype(str).str.lower().isin(["yes", "1", "true"]).astype(int)

# ---------- 3. Encoding ----------
enc = {}
for c in ["Area_Name", "Road_Name", "Weather_Condition"]:
    enc[c] = LabelEncoder().fit(df[c].astype(str)); df[c + "_Enc"] = enc[c].transform(df[c].astype(str))
FEATURES = ["Area_Name_Enc", "Road_Name_Enc", "Hour", "Day", "Month", "DayOfWeek", "Is_Weekend", "Peak_Hour",
            "Hour_Sin", "Hour_Cos", "Weather_Condition_Enc", "Incident_Reports", "Roadwork_Flag",
            "Traffic_Signal_Compliance", "Parking_Usage", "Public_Transport_Usage",
            "Pedestrian_and_Cyclist_Count", "Environmental_Impact"]
y = df["Congestion_Level"].map({c: i for i, c in enumerate(CLASSES)})

def models():
    return {"Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric="mlogloss", random_state=42)}

# ---------- 4. LEAKAGE CHECK ----------
leak = {"dropped_columns": [c for c in LEAKY if c in df.columns]}
if leak["dropped_columns"]:
    Xl = df[FEATURES + leak["dropped_columns"]]
    a, b, ya, yb = train_test_split(Xl, y, test_size=0.2, stratify=y, random_state=42)
    leak["accuracy_WITH_leaky_columns"] = round(float(accuracy_score(yb, RandomForestClassifier(100, random_state=42, n_jobs=-1).fit(a, ya).predict(b))), 4)
    single = {}
    for c in leak["dropped_columns"]:   # how well does ONE column alone predict the target?
        a1, b1, ya1, yb1 = train_test_split(df[[c]], y, test_size=0.2, stratify=y, random_state=42)
        single[c] = round(float(accuracy_score(yb1, DecisionTreeClassifier(max_depth=4).fit(a1, ya1).predict(b1))), 4)
    leak["single_column_accuracy"] = single
    print("LEAKAGE CHECK:", leak)

# ---------- 5. Split + train + evaluate (leak-free features only) ----------
X = df[FEATURES]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
results, trained, cms = {}, {}, {}
for name, m in models().items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="weighted", zero_division=0)
    results[name] = {"accuracy": round(accuracy_score(yte, p), 4), "precision": round(pr, 4),
                     "recall": round(rc, 4), "f1": round(f1, 4)}
    cms[name] = confusion_matrix(yte, p, labels=[0, 1, 2, 3]); trained[name] = m
    print(name, results[name])
best = max(results, key=lambda k: results[k]["f1"]); print("Best model:", best)
leak["accuracy_WITHOUT_leaky_columns_best"] = results[best]["accuracy"]

# ---------- 6. Charts / artifacts ----------
for name, cm in cms.items():
    plt.figure(figsize=(5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Confusion Matrix - {name}"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
    plt.savefig(f"static/plots/cm_{name.replace(' ', '_').lower()}.png", dpi=110); plt.close()

imp = pd.Series(trained[best].feature_importances_, index=FEATURES).sort_values(ascending=False)
pretty = lambda s: s.replace("_Enc", "").replace("_", " ")
hourly = df.groupby("Hour")["Traffic_Volume"].mean().round(0) if "Traffic_Volume" in df else df.groupby("Hour").size()
stats = {
    "rows": int(len(df)), "areas": int(df["Area_Name"].nunique()), "roads": int(df["Road_Name"].nunique()),
    "avg_speed": round(float(df["Average_Speed"].mean()), 1) if "Average_Speed" in df else None,
    "avg_volume": int(df["Traffic_Volume"].mean()) if "Traffic_Volume" in df else None,
    "distribution": df["Congestion_Level"].value_counts().reindex(CLASSES).fillna(0).astype(int).tolist(),
    "hourly_labels": [int(h) for h in hourly.index], "hourly_values": [float(v) for v in hourly.values],
    "area_severity": (df.assign(s=y).groupby("Area_Name")["s"].mean().round(2)).to_dict(),
}
metrics = {"models": results, "best_model": best, "classes": CLASSES, "leakage": leak,
           "importance": {pretty(k): round(float(v), 4) for k, v in imp.items()},
           "confusion": {k: v.tolist() for k, v in cms.items()}, "stats": stats,
           "note": "Trained on historical data. Not a real-time system."}
json.dump(metrics, open("models/metrics.json", "w"), indent=2)

# Typical values + road lookup used by the web app (defaults, occlusion explanations, route suggestions)
meta = {"features": FEATURES, "classes": CLASSES, "enc": {k: list(v.classes_) for k, v in enc.items()},
        "defaults": {f: float(X[f].median()) for f in FEATURES},
        "roads": df.groupby("Area_Name")["Road_Name"].unique().apply(list).to_dict(),
        "road_area": df.drop_duplicates("Road_Name").set_index("Road_Name")["Area_Name"].to_dict(),
        "weather_options": sorted(df["Weather_Condition"].astype(str).unique().tolist())}
joblib.dump({"model": trained[best], "encoders": enc, "meta": meta}, "models/best_model.pkl")
print("Saved models/best_model.pkl and models/metrics.json")
