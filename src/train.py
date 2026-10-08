"""Train all six detectors on NORMAL traffic only, compare, and save the best.

Run from the project root:
    python -m src.train                                  # synthetic data
    python -m src.train --normal-csv normal_flows.csv    # your own captured normal flows
"""
import argparse, io, os, time
import numpy as np, joblib
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.common import Detector, simulate

MODELS = {
    "OCSVM": lambda: OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"),
    "LOF":   lambda: LocalOutlierFactor(n_neighbors=20, novelty=True),
    "KDE":   lambda: KernelDensity(kernel="gaussian", bandwidth=0.5),
    "GMM":   lambda: GaussianMixture(n_components=4, covariance_type="full", random_state=0),
    "MCD":   lambda: MinCovDet(support_fraction=0.9, random_state=0),
    "IsolationForest": lambda: IsolationForest(n_estimators=100, random_state=0),
}
ATTACKS = ["dos", "scan", "exfil", "unknown"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal-csv", default=None)
    ap.add_argument("--fpr", type=float, default=5.0, help="%% of normal val flows allowed above threshold")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    normal = (np.loadtxt(a.normal_csv, delimiter=",", skiprows=1)
              if a.normal_csv else simulate("normal", 5000, seed=1))
    n = len(normal)
    tr, val, te_norm = normal[:int(.6 * n)], normal[int(.6 * n):int(.8 * n)], normal[int(.8 * n):]

    X_test = np.vstack([te_norm] + [simulate(k, 250, seed=10 + i) for i, k in enumerate(ATTACKS)])
    y_test = np.r_[np.zeros(len(te_norm)), np.ones(250 * len(ATTACKS))]
    unknown_mask = np.r_[np.zeros(len(te_norm) + 250 * 3, bool), np.ones(250, bool)]

    scaler = StandardScaler().fit(Detector.prep(tr))
    rows, best, best_val = [], None, -1e9
    print(f"{'Model':16}{'F1':>7}{'Prec':>7}{'Rec':>7}{'AUC':>7}{'Unk.Rec':>9}{'ms/flow':>9}{'KB':>8}")
    for name, make in MODELS.items():
        t0 = time.time()
        det = Detector(name, make().fit(scaler.transform(Detector.prep(tr))), scaler)
        train_s = time.time() - t0
        det.threshold = float(np.percentile(det.score(val), 100 - a.fpr))

        s = det.score(X_test)
        pred = (s > det.threshold).astype(int)
        t0 = time.perf_counter()
        for x in X_test[:200]:
            det.predict(x)
        ms = (time.perf_counter() - t0) / 200 * 1000
        buf = io.BytesIO(); joblib.dump(det, buf); kb = buf.tell() / 1024

        r = dict(model=name, f1=f1_score(y_test, pred), precision=precision_score(y_test, pred),
                 recall=recall_score(y_test, pred), auc=roc_auc_score(y_test, s),
                 unknown_recall=pred[unknown_mask].mean(), ms_per_flow=ms, size_kb=kb, train_s=train_s)
        rows.append(r)
        print(f"{name:16}{r['f1']:7.3f}{r['precision']:7.3f}{r['recall']:7.3f}{r['auc']:7.3f}"
              f"{r['unknown_recall']:9.3f}{ms:9.2f}{kb:8.0f}")
        value = r["f1"] - 0.02 * ms  # accuracy vs. latency trade-off (edit to taste)
        if value > best_val:
            best, best_val = det, value

    with open(os.path.join(a.out, "results.csv"), "w") as f:
        f.write(",".join(rows[0]) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    joblib.dump(best, os.path.join(a.out, "best_model.joblib"), compress=3)
    print(f"\nSelected: {best.name}  (saved to {a.out}/best_model.joblib)")


if __name__ == "__main__":
    main()
