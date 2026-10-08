"""Flask API deployed on Vercel. Serves the demo dashboard's predictions."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib
from flask import Flask, jsonify, request
from src.common import FEATURES, PROFILES, simulate

app = Flask(__name__)
det = joblib.load(os.path.join(ROOT, "models", "best_model.joblib"))


def verdict(x):
    s = float(det.score(x)[0])
    return {"features": dict(zip(FEATURES, [round(float(v), 4) for v in x])),
            "score": round(s, 4), "threshold": round(float(det.threshold), 4),
            "status": "ANOMALY DETECTED" if s > det.threshold else "NORMAL"}


@app.get("/api/info")
def info():
    return jsonify(model=det.name, features=FEATURES, kinds=list(PROFILES))


@app.get("/api/simulate")
def sim():
    kind = request.args.get("kind", "normal")
    if kind not in PROFILES:
        return jsonify(error="unknown kind"), 400
    return jsonify(kind=kind, **verdict(simulate(kind, 1)[0]))


@app.post("/api/predict")
def predict():
    data = request.get_json(force=True, silent=True) or {}
    x = data.get("features")
    if not isinstance(x, list) or len(x) != len(FEATURES):
        return jsonify(error=f"send 'features' as a list of {len(FEATURES)} numbers: {FEATURES}"), 400
    return jsonify(verdict([float(v) for v in x]))
