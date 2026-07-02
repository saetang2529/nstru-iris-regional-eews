#!/usr/bin/env python
# NOTE: adjust the input/output paths below to your own measurement files.
# mbP PER-STATION CALIBRATION (2026-06-13) — fit a fast P-window body-wave magnitude to USGS mb, per station.
# Joint least-squares (zero-mean station terms):  USGS_mb ~ log10(pd10cm) + a*log10(Rkm) + b + station_term
# Anchored ONLY on USGS mb-type events from the Step-F harvest (the consistent same-unit truth). Research only — no live changes.
# Output: per-station calibration JSON + in-sample AND 80/20 held-out validation + per-magnitude-bin behaviour.
import json, numpy as np, collections
np.random.seed(7)
M = json.load(open("./stepF_measurements.json"))
EV, MEAS = M["events"], M["measurements"]

rows = []   # (eid, sta, Rkm, pd10, usgs_mb)
for m in MEAS:
    if len(m) < 8:
        continue
    eid, sta, band, Rkm, snr10, mint_w, pd3, pd10 = m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]
    ev = EV.get(eid)
    if not ev:
        continue
    if not str(ev.get("magType", "")).lower().startswith("mb"):   # mb anchor only (same-unit)
        continue
    if not (pd10 and pd10 > 0) or not (snr10 and snr10 >= 2.5):   # match the live SNR10>=2.5 gate
        continue
    rows.append((eid, sta, float(Rkm), float(pd10), float(ev["M"])))
print("mb pairs (snr10>=2.5, pd10>0):", len(rows))

def fit(train):
    stations = sorted(set(r[1] for r in train))
    sidx = {s: i for i, s in enumerate(stations)}
    ns = len(stations); N = len(train)
    y = np.array([r[4] - np.log10(r[3]) for r in train])          # mb - log10(pd10) = a*log10(R)+b+s_j
    X = np.zeros((N, 2 + ns))
    for i, r in enumerate(train):
        X[i, 0] = np.log10(r[2]); X[i, 1] = 1.0; X[i, 2 + sidx[r[1]]] = 1.0
    C = np.zeros((1, 2 + ns)); C[0, 2:] = 1.0                     # zero-mean station-term constraint
    coef, *_ = np.linalg.lstsq(np.vstack([X, 500.0 * C]), np.concatenate([y, [0.0]]), rcond=None)
    a, b = coef[0], coef[1]
    sterm = {stations[i]: float(coef[2 + i]) for i in range(ns)}
    return a, b, sterm

def event_pred(rs, a, b, sterm):
    vals = [np.log10(r[3]) + a * np.log10(r[2]) + b + sterm.get(r[1], 0.0) for r in rs]
    return float(np.median(vals))

def report(tag, test, a, b, sterm):
    byev = collections.defaultdict(list)
    for r in test:
        byev[r[0]].append(r)
    errs, mags = [], []
    for eid, rs in byev.items():
        e = event_pred(rs, a, b, sterm) - rs[0][4]
        errs.append(e); mags.append(rs[0][4])
    errs = np.array(errs); mags = np.array(mags)
    print("%s: n=%d  bias %+.3f  MAE %.3f  within +-0.3: %.0f%%" %
          (tag, len(errs), np.mean(errs), np.mean(np.abs(errs)), 100 * np.mean(np.abs(errs) <= 0.3)))
    for lo, hi in [(4.0, 4.5), (4.5, 5.0), (5.0, 5.5), (5.5, 6.6)]:
        s = errs[(mags >= lo) & (mags < hi)]
        if len(s):
            print("    M%.1f-%.1f: n=%4d bias %+.3f MAE %.3f" % (lo, hi, len(s), np.mean(s), np.mean(np.abs(s))))
    return errs

# in-sample
a, b, sterm = fit(rows)
print("\nfit:  a(dist)=%.3f  b(const)=%.3f  stations=%d" % (a, b, len(sterm)))
report("IN-SAMPLE  mbP vs USGS mb", rows, a, b, sterm)

# 80/20 held-out by EVENT (honest)
eids = sorted(set(r[0] for r in rows))
np.random.shuffle(eids)
cut = int(0.8 * len(eids)); train_e = set(eids[:cut]); test_e = set(eids[cut:])
tr = [r for r in rows if r[0] in train_e]; te = [r for r in rows if r[0] in test_e]
a2, b2, st2 = fit(tr)
print()
report("HELD-OUT 20%  mbP vs USGS mb", te, a2, b2, st2)

# compare: what does the RAW MwP-style do here? (sanity: pd10 alone with no calib)
json.dump({"a": a, "b": b, "station_term": sterm,
           "form": "mbP = log10(pd10cm) + a*log10(Rkm) + b + station_term, anchored to USGS mb, snr10>=2.5",
           "n_pairs": len(rows), "n_stations": len(sterm)},
          open("./mbp_calib.json", "w"))
print("\nsaved -> mbp_calib.json")
print("DONE")
