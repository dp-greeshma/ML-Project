# Smart Traffic Congestion Prediction (Explainable ML)

## Folder structure
traffic_project/
├── app.py                    Flask server: 6 pages + /api/predict, explanations, route suggestions
├── train_model.py            Cleaning, leakage check, feature engineering, trains 3 models, saves model + charts
├── generate_sample_data.py   OPTIONAL: makes a synthetic demo dataset (use only if you lack the real CSV)
├── requirements.txt
├── data/traffic_data.csv     Your dataset goes here
├── models/                   best_model.pkl, metrics.json (created by train_model.py)
├── static/css/style.css      Styling   | static/js/main.js  Charts + form logic | static/plots/ confusion matrices
└── templates/                base, index, predict, performance, analysis, routes, about (HTML pages)

## Run locally
    pip install -r requirements.txt
    # put your dataset at data/traffic_data.csv  (or: python generate_sample_data.py)
    python train_model.py
    python app.py            # open http://127.0.0.1:5000

## Run in Google Colab
    !pip install -q flask flask-ngrok pyngrok xgboost seaborn
    # upload the project folder (or unzip it), then:
    %cd traffic_project
    !python train_model.py
    from pyngrok import ngrok            # free token from ngrok.com
    ngrok.set_auth_token("YOUR_TOKEN")
    import threading, app
    threading.Thread(target=lambda: app.app.run(port=5000)).start()
    print(ngrok.connect(5000))

## Using YOUR dataset
Column names are normalised automatically (spaces and "/" become "_"). Required: Area Name, Road/Intersection Name,
Weather Condition, Incident Reports, Roadwork and Construction Activity, Traffic Signal Compliance, Parking Usage,
Public Transport Usage, Pedestrian and Cyclist Count, Environmental Impact, Hour/Day/Month/DayOfWeek (or a Date column),
Congestion_Level (text classes, or numeric which is split into 4 quartile classes).
If your target column is named differently, edit the `ren` dict in train_model.py.
Check the leakage section printed by train_model.py: add any other column that mirrors the target to the LEAKY list.
