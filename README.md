# Lightweight ML Anomaly Intrusion Detection System for IoMT Networks

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?logo=vercel)](https://YOUR-PROJECT.vercel.app)

🔗 **Live demo:** [https://YOUR-PROJECT.vercel.app](https://io-tm-anamoly-ids.vercel.app/)

An anomaly/novelty-based IDS that learns *normal* behaviour of Internet-of-Medical-Things traffic and flags
deviations, including unknown attacks. Six detectors are compared (OCSVM, LOF, KDE, GMM, MCD, Isolation Forest)
and the best accuracy/latency trade-off is deployed on a Raspberry Pi.

## Architecture
```
Packets -> Sniffer (scapy) -> Flow builder -> 8 features -> Scaler -> Model -> Score > threshold?
                                                                   |-> NORMAL
                                                                   |-> ANOMALY / INTRUSION ALERT (+ alerts.log, status.json)
```

## Quick start
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-pi.txt
python -m src.train                      # trains 6 models on normal traffic, saves models/best_model.joblib
sudo python -m src.realtime --iface wlan0    # live monitoring (Raspberry Pi / Linux)
```
Use your own data: record normal traffic with `sudo python -m src.realtime --iface wlan0 --collect normal_flows.csv`,
then `python -m src.train --normal-csv normal_flows.csv`.

## Project layout
| Path | Purpose |
|---|---|
| `src/common.py` | features, traffic simulator, `Detector` wrapper |
| `src/train.py` | trains and benchmarks all six models, picks the best |
| `src/realtime.py` | live sniffer + detection + alerts (Raspberry Pi) |
| `api/index.py`, `public/index.html` | web demo deployed on Vercel |
| `models/` | `best_model.joblib`, `results.csv` |

## Results
See `models/results.csv` (F1, AUC, recall on held-out *unknown* attacks, latency per flow, model size).

## Notes
The Vercel site is a **demo** that scores simulated flows. Packet capture runs on the Pi, not on Vercel.
Train and deploy with the same `scikit-learn` version (pinned in `requirements.txt`).
