"""
app.py - Flask web server.
Loads the trained model (models/best_model.pkl) and serves 6 pages + 2 JSON APIs.
NOTE: predictions use HISTORICAL patterns. This is NOT real-time traffic prediction.
"""
import json
from datetime import datetime
import numpy as np, pandas as pd, joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
bundle = joblib.load("models/best_model.pkl")
MODEL, ENC, META = bundle["model"], bundle["encoders"], bundle["meta"]
METRICS = json.load(open("models/metrics.json"))
CLASSES, FEATURES, DEFAULTS = META["classes"], META["features"], META["defaults"]
WEIGHTS = np.array([0, 1, 2, 3])          # Low..Severe -> risk score used for ranking/explanations

# Which model features belong to which user-facing input (used for explanations)
GROUPS = {
    "Location": ["Area_Name_Enc", "Road_Name_Enc"],
    "Time of day": ["Hour", "Hour_Sin", "Hour_Cos", "Peak_Hour"],
    "Day of week": ["DayOfWeek", "Is_Weekend"],
    "Weather": ["Weather_Condition_Enc"],
    "Incident reports": ["Incident_Reports"],
    "Roadwork": ["Roadwork_Flag"],
    "Signal compliance": ["Traffic_Signal_Compliance"],
    "Parking usage": ["Parking_Usage"],
    "Public transport usage": ["Public_Transport_Usage"],
    "Pedestrian/cyclist count": ["Pedestrian_and_Cyclist_Count"],
    "Environmental impact": ["Environmental_Impact"],
}

def build_row(inp):
    """Turn form inputs into a one-row DataFrame with the exact training features."""
    row = dict(DEFAULTS)
    road = inp.get("road") or META["roads"][list(META["roads"])[0]][0]
    area = META["road_area"].get(road, list(META["roads"])[0])
    hour, dow = int(inp.get("hour", 8)), int(inp.get("day_of_week", 0))
    row.update({
        "Area_Name_Enc": ENC["Area_Name"].transform([area])[0],
        "Road_Name_Enc": ENC["Road_Name"].transform([road])[0],
        "Hour": hour, "DayOfWeek": dow, "Is_Weekend": int(dow >= 5),
        "Peak_Hour": int(hour in (7, 8, 9, 10, 17, 18, 19, 20)),
        "Hour_Sin": np.sin(2 * np.pi * hour / 24), "Hour_Cos": np.cos(2 * np.pi * hour / 24),
        "Weather_Condition_Enc": ENC["Weather_Condition"].transform([inp.get("weather", META["weather_options"][0])])[0],
        "Incident_Reports": float(inp.get("incidents", 0)),
        "Roadwork_Flag": int(str(inp.get("roadwork", "No")).lower() in ("yes", "1", "true")),
        "Traffic_Signal_Compliance": float(inp.get("compliance", DEFAULTS["Traffic_Signal_Compliance"])),
        "Parking_Usage": float(inp.get("parking", DEFAULTS["Parking_Usage"])),
        "Public_Transport_Usage": float(inp.get("pt_usage", DEFAULTS["Public_Transport_Usage"])),
        "Pedestrian_and_Cyclist_Count": float(inp.get("pedestrians", DEFAULTS["Pedestrian_and_Cyclist_Count"])),
        "Environmental_Impact": float(inp.get("environment", DEFAULTS["Environmental_Impact"])),
    })
    return pd.DataFrame([row])[FEATURES]

def probs_of(df):
    return MODEL.predict_proba(df)

def alert_for(level, p):
    msgs = {"Low": ("success", "Traffic expected to flow normally."),
            "Medium": ("warning", "Moderate traffic. Allow some extra travel time."),
            "High": ("danger", "Heavy congestion likely. Consider an alternative route."),
            "Severe": ("critical", "Severe congestion likely. Avoid this road if possible.")}
    kind, text = msgs[level]
    return {"type": kind, "message": text, "confidence": round(float(max(p)) * 100, 1)}

def explain(row, base_score):
    """Occlusion explanation: reset one input group to a typical value and see how the risk score changes."""
    out = []
    for name, cols in GROUPS.items():
        r2 = row.copy()
        for c in cols: r2[c] = DEFAULTS[c]
        s2 = float(probs_of(r2)[0] @ WEIGHTS)
        out.append({"factor": name, "effect": round(base_score - s2, 3)})
    out.sort(key=lambda d: abs(d["effect"]), reverse=True)
    return out[:6]

def suggest_routes(inp, current_score):
    """Compare other roads under the same time/weather. Alternatives assume no road-specific incident/roadwork."""
    cur = inp.get("road"); cur_area = META["road_area"].get(cur)
    cands = []
    for road, area in META["road_area"].items():
        if road == cur: continue
        alt = dict(inp, road=road, incidents=0, roadwork="No")
        p = probs_of(build_row(alt))[0]; s = float(p @ WEIGHTS)
        if s < current_score - 0.2:
            cands.append({"road": road, "area": area, "same_area": area == cur_area,
                          "level": CLASSES[int(p.argmax())], "risk_score": round(s, 2),
                          "improvement": round((current_score - s) / max(current_score, 0.01) * 100)})
    cands.sort(key=lambda d: (not d["same_area"], d["risk_score"]))
    return cands[:3]

def predict_core(inp):
    row = build_row(inp); p = probs_of(row)[0]
    level = CLASSES[int(p.argmax())]; score = float(p @ WEIGHTS)
    res = {"level": level, "probabilities": {c: round(float(v) * 100, 1) for c, v in zip(CLASSES, p)},
           "alert": alert_for(level, p), "factors": explain(row, score), "routes": []}
    if level in ("High", "Severe"):
        res["routes"] = suggest_routes(inp, score)
    return res

def dashboard_snapshot():
    """Scenario for the CURRENT clock hour using typical (median) conditions across all monitored roads.
    This is NOT live data."""
    now = datetime.now(); rows, names = [], []
    for road in META["road_area"]:
        rows.append(build_row({"road": road, "hour": now.hour, "day_of_week": now.weekday(), "weather": META["weather_options"][0]}))
        names.append(road)
    P = probs_of(pd.concat(rows)); avg = P.mean(axis=0); scores = P @ WEIGHTS
    order = np.argsort(-scores)[:4]
    alerts = [{"road": names[i], "level": CLASSES[int(P[i].argmax())], "score": round(float(scores[i]), 2)} for i in order]
    return {"hour": now.hour, "level": CLASSES[int(avg.argmax())],
            "probabilities": {c: round(float(v) * 100, 1) for c, v in zip(CLASSES, avg)}, "alerts": alerts}

def form_options():
    return dict(roads=META["roads"], weather=META["weather_options"], defaults=DEFAULTS)

@app.route("/")
def home():
    return render_template("index.html", snap=dashboard_snapshot(), m=METRICS, active="home")

@app.route("/predict")
def predict_page():
    return render_template("predict.html", opts=form_options(), active="predict")

@app.route("/performance")
def performance():
    return render_template("performance.html", m=METRICS, active="performance")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html", m=METRICS, active="analysis")

@app.route("/routes")
def routes_page():
    return render_template("routes.html", opts=form_options(), active="routes")

@app.route("/about")
def about():
    return render_template("about.html", m=METRICS, active="about")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        return jsonify(predict_core(request.get_json(force=True)))
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

if __name__ == "__main__":
    app.run(debug=True)
