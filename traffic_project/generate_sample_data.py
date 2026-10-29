"""
generate_sample_data.py
-----------------------
Creates a SYNTHETIC demo dataset (data/traffic_data.csv) so the project runs
even if you do not have the real dataset. If you have your own CSV, simply
save it as data/traffic_data.csv and skip this script.
NOTE: synthetic data -> results are for demonstration only.
"""
import os, numpy as np, pandas as pd
os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(42)
N = 8000
areas = {
    "Indiranagar": ["100 Feet Road", "CMH Road", "Sony World Junction"],
    "Koramangala": ["Sony World Signal", "Intermediate Ring Road", "80 Feet Road"],
    "Whitefield": ["Marathahalli Bridge", "ITPL Main Road", "Hope Farm Junction"],
    "Jayanagar": ["South End Circle", "30th Cross Road", "Jayanagar 4th Block"],
    "Hebbal": ["Hebbal Flyover", "Ballari Road", "Outer Ring Road"],
}
rows = []
for _ in range(N):
    a = rng.choice(list(areas)); r = rng.choice(areas[a])
    d = pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 730)))
    hour = int(rng.integers(0, 24))
    peak = int(hour in (7, 8, 9, 10, 17, 18, 19, 20))
    weather = rng.choice(["Clear", "Overcast", "Fog", "Rain", "Windy"], p=[.4, .25, .08, .22, .05])
    incidents = int(rng.poisson(0.8 + 0.6 * peak))
    roadwork = rng.choice(["Yes", "No"], p=[.2, .8])
    compliance = float(np.clip(rng.normal(75, 12), 30, 100))
    parking = float(np.clip(rng.normal(60 + 12 * peak, 18), 5, 100))
    pt = float(np.clip(rng.normal(50, 15), 5, 100))
    ped = int(np.clip(rng.normal(150 + 80 * peak, 60), 5, 600))
    env = float(np.clip(rng.normal(80 + 25 * peak, 20), 20, 200))
    score = (2.2 * peak + 0.5 * incidents + 0.9 * (roadwork == "Yes")
             + {"Clear": 0, "Overcast": .2, "Fog": .6, "Rain": 1.0, "Windy": .1}[weather]
             + 0.02 * (parking - 60) - 0.02 * (compliance - 75) - 0.01 * (pt - 50)
             + 0.002 * (ped - 150) + rng.normal(0, 0.9))
    # These 4 columns are computed FROM congestion -> they would LEAK the target
    volume = int(np.clip(25000 + 5500 * score + rng.normal(0, 1500), 3000, 70000))
    speed = float(np.clip(55 - 6 * score + rng.normal(0, 2), 5, 70))
    tti = float(np.clip(1.0 + 0.18 * (score + 2) + rng.normal(0, .03), 1, 3))
    util = float(np.clip(50 + 11 * score + rng.normal(0, 2), 10, 100))
    rows.append([d, a, r, volume, speed, tti, util, incidents, env, pt, compliance, parking,
                 ped, weather, roadwork, hour, d.day, d.month, d.dayofweek, peak, score])
cols = ["Date", "Area Name", "Road/Intersection Name", "Traffic Volume", "Average Speed",
        "Travel Time Index", "Road Capacity Utilization", "Incident Reports", "Environmental Impact",
        "Public Transport Usage", "Traffic Signal Compliance", "Parking Usage",
        "Pedestrian and Cyclist Count", "Weather Condition", "Roadwork and Construction Activity",
        "Hour", "Day", "Month", "DayOfWeek", "Peak_Hour", "_score"]
df = pd.DataFrame(rows, columns=cols)
df["Congestion_Level"] = pd.cut(df["_score"], [-99, 1.5, 3.0, 4.5, 99],
                                labels=["Low", "Medium", "High", "Severe"])
df = df.drop(columns="_score")
df.to_csv("data/traffic_data.csv", index=False)
print("Saved data/traffic_data.csv", df.shape)
print(df["Congestion_Level"].value_counts())
