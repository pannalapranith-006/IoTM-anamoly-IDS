"""Real-time IoMT traffic monitor (run on the Raspberry Pi / gateway, needs root).

    sudo python -m src.realtime --iface wlan0                              # detect
    sudo python -m src.realtime --iface wlan0 --collect normal_flows.csv   # record NORMAL traffic for training
"""
import argparse, csv, json, logging, time
from collections import defaultdict

import joblib, numpy as np
from scapy.all import IP, TCP, UDP, sniff

from src.common import FEATURES


def flows_from(packets):
    flows = defaultdict(list)
    for p in packets:
        if IP not in p:
            continue
        if TCP in p:
            proto, sp, dp, fl = 1, p[TCP].sport, p[TCP].dport, int(p[TCP].flags)
        elif UDP in p:
            proto, sp, dp, fl = 2, p[UDP].sport, p[UDP].dport, 0
        else:
            proto, sp, dp, fl = 0, 0, 0, 0
        flows[(p[IP].src, p[IP].dst, sp, dp, proto)].append((float(p.time), len(p), fl))

    keys, rows = [], []
    for key, pk in flows.items():
        t = np.array([x[0] for x in pk]); s = np.array([x[1] for x in pk], float)
        dur = max(t.max() - t.min(), 1e-3)
        syn = np.mean([bool(x[2] & 0x02) and not bool(x[2] & 0x10) for x in pk])
        keys.append(key)
        rows.append([dur, len(pk), s.sum(), s.mean(), s.std(), dur / len(pk), syn, key[4]])
    return keys, np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default=None)
    ap.add_argument("--window", type=float, default=5.0, help="seconds per analysis window")
    ap.add_argument("--model", default="models/best_model.joblib")
    ap.add_argument("--collect", default=None, help="save flows of NORMAL traffic to this CSV")
    ap.add_argument("--alert-streak", type=int, default=3)
    a = ap.parse_args()

    logging.basicConfig(filename="alerts.log", level=logging.INFO, format="%(asctime)s %(message)s")
    det = None if a.collect else joblib.load(a.model)
    if a.collect:
        with open(a.collect, "w", newline="") as f:
            csv.writer(f).writerow(FEATURES)

    streak = 0
    print("Monitoring... Ctrl+C to stop")
    while True:
        pkts = sniff(iface=a.iface, timeout=a.window, store=True)
        keys, X = flows_from(pkts)
        if len(X) == 0:
            continue
        if a.collect:
            with open(a.collect, "a", newline="") as f:
                csv.writer(f).writerows(X.tolist())
            print(f"collected {len(X)} flows")
            continue

        scores = det.score(X)
        bad = scores > det.threshold
        if bad.any():
            streak += 1
            status = "INTRUSION ALERT" if streak >= a.alert_streak else "ANOMALY DETECTED (possible unknown attack)"
            worst = keys[int(np.argmax(scores))]
            msg = f"{status} | {int(bad.sum())}/{len(X)} flows | worst={worst} score={scores.max():.2f}"
            print(msg); logging.warning(msg)
        else:
            streak, status = 0, "NORMAL"
            print(f"{time.strftime('%H:%M:%S')} NORMAL | {len(X)} flows")
        with open("status.json", "w") as f:  # read by a dashboard / other services
            json.dump({"time": time.time(), "status": status, "flows": len(X),
                       "anomalous_flows": int(bad.sum())}, f)


if __name__ == "__main__":
    main()
