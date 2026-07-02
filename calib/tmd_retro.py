#!/usr/bin/env python
# NOTE: adjust the input/output paths below to your own measurement files.
# TMD FULL-CATALOG RETROSPECTIVE — built 2026-06-13.
# Q: how does our CURRENT system (Step-E tree, md5 82494f38) perform on the TMD catalog?
#   - epicentre error vs TMD location (trusted), origin-time error vs TMD time (trusted),
#   - magnitude: our MwP vs the TMD catalog magnitude (TMD ML is UNCALIBRATED — reference only), by magnitude bin.
# TMD time+location = truth; TMD magnitude = reference only. Research only — NO live changes.
# r_detect pre-filter: events with <4 stations inside the magnitude's detectability radius are classified
#   "too_few_stations" WITHOUT a wasteful waveform fetch (most of the 9,302 M<3 events).
# Usage: python tmd_retro.py [--minmag X] [--limit N] [--out NAME]   (no args = full catalog)
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, time, math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from obspy import Stream, UTCDateTime, read_inventory

PROJ = "./projE"                 # Step-E code tree (md5 82494f38)
sys.path.insert(0, PROJ); os.chdir(PROJ); sys.argv = ["regional_eews.py"]
import regional_eews as R
R.log = lambda m: None
R.build_station_tables(); R._ensure_tt()
try:
    import scipy.signal.windows._windows as _sw
    if "hanning" not in _sw._win_equiv: _sw._win_equiv["hanning"] = _sw._win_equiv["hann"]
except Exception: pass
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel
TAUP = TauPyModel("iasp91"); NWORK = 11

# ---- args ----
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--minmag", type=float, default=0.0)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="tmd_retro_results.json")
A = ap.parse_args()

CATF = "./tmd_catalog.json"
OUTDIR = "./out"
OUT = os.path.join(OUTDIR, A.out)
INV2 = R.INV + read_inventory("./tm_inv_response.xml")   # cached TM responses (no re-fetch)

R_ANCHORS = [(1.5, 30.0), (2.0, 50.0), (2.5, 80.0), (3.0, 150.0),
             (4.0, 300.0), (5.0, 600.0), (6.0, 1200.0), (7.0, 2500.0), (8.0, 5000.0)]
def r_detect(mag):
    ms = [a[0] for a in R_ANCHORS]
    if mag <= ms[0]: return R_ANCHORS[0][1]
    if mag >= ms[-1]: return R_ANCHORS[-1][1]
    for i in range(len(ms) - 1):
        if ms[i] <= mag <= ms[i + 1]:
            f = (mag - ms[i]) / (ms[i + 1] - ms[i])
            return 10 ** (math.log10(R_ANCHORS[i][1]) + f * (math.log10(R_ANCHORS[i + 1][1]) - math.log10(R_ANCHORS[i][1])))
    return R_ANCHORS[-1][1]

cat = json.load(open(CATF))
cat = [e for e in cat if e["mag"] >= A.minmag]
if A.limit: cat = cat[-A.limit:]                  # newest N (best data coverage)
print("TMD events to test: %d (minmag=%.1f limit=%d)" % (len(cat), A.minmag, A.limit), flush=True)

tasks = []; evmeta = {}; classify = {}; n_nofetch = 0
for ev in cat:
    elat, elon = ev["lat"], ev["lon"]; edep = ev["dep"] if ev["dep"] else 10.0
    mag = ev["mag"]; otime = UTCDateTime(ev["utc"])
    rmax = min(r_detect(mag), 300.0)              # detectability radius, capped at 300 km (fetch feasibility)
    near = []
    for s, c in R.STATION_COORDS.items():
        Rkm = gps2dist_azimuth(elat, elon, c["lat"], c["lon"])[0] / 1000.0
        if 10.0 <= Rkm <= rmax:
            near.append((Rkm, s))
    eid = "%s_M%.1f_%s" % (str(otime)[:19], mag, ev.get("id", "?"))
    evmeta[eid] = {"time": str(otime), "M": float(mag), "magType": "TMD-ML",
                   "lat": elat, "lon": elon, "dep": edep, "region": ev.get("region", "")}
    if len(near) < 4:
        classify[eid] = "too_few_stations(%d in %.0fkm)" % (len(near), rmax); n_nofetch += 1
        continue
    classify[eid] = "candidate(%d stn)" % len(near)
    for Rkm, s in sorted(near)[:6]:
        nt, lc, cz, ig = R._STA_SRC.get(s, (R.STATION_NET.get(s, ""), "", "HHZ", False))
        chans = ["HHZ", "BHZ"] if nt == "TM" else [cz]
        d = locations2degrees(elat, elon, R.STATION_COORDS[s]["lat"], R.STATION_COORDS[s]["lon"])
        try:
            arr = TAUP.get_travel_times(source_depth_in_km=min(edep, 700), distance_in_degree=d, phase_list=["P", "Pn", "Pg", "p"])
        except Exception:
            continue
        if not arr: continue
        ref = otime + min(a.time for a in arr) + 10.0
        tasks.append((eid, s, nt, chans, lc, ig, ref, Rkm))
print("candidates(>=4 stn in range): %d | skipped(no fetch): %d | fetch tasks: %d"
      % (len(evmeta) - n_nofetch, n_nofetch, len(tasks)), flush=True)

def work(task):
    eid, s, nt, chans, lc, ig, ref, Rkm = task
    try:
        st = None; band = None
        for cz in chans:
            try:
                loc = "*" if nt == "TM" else (lc if lc else "*")
                x = R._fdsn(ig).get_waveforms(nt, s, loc, cz, ref - 55, ref + 40)
            except Exception:
                continue
            if x and len(x): st = x; band = cz[:2]; break
        if st is None: return None
        tr = st.select(component="Z").merge(method=1, fill_value=0)[0]
        if abs(tr.stats.sampling_rate - R.SAMPLE_RATE) > 0.01: tr.resample(R.SAMPLE_RATE)
        dt = tr.stats.delta
        p_utc, cft = R.pick_p_arrival(Stream([tr.slice(ref - R.PICK_WINDOW_SEC, ref)]), ref)
        if cft <= 0: return None
        vel = tr.copy().remove_response(inventory=INV2, output="VEL", pre_filt=(0.005, 0.01, 40, 45))
        hfv = vel.copy().filter("highpass", freq=1.0, corners=4, zerophase=True)
        noi = hfv.slice(p_utc - 40, p_utc - 5).data; sig = hfv.slice(p_utc, p_utc + 10).data
        if len(noi) < 10 or len(sig) < 10: return None
        snr = float(np.sqrt(np.mean(sig ** 2)) / (np.sqrt(np.mean(noi ** 2)) + 1e-12))
        if snr < 2.0: return None
        disp = tr.copy().remove_response(inventory=INV2, output="DISP", pre_filt=(0.01, 0.02, 40, 45))
        sg10 = disp.slice(p_utc, p_utc + 10).data
        if len(sg10) <= 20: return None
        dm = sg10 - np.mean(sg10); cum = np.cumsum(dm) * dt
        mint_w = {}
        for w in range(1, 11):
            n = int(w / dt)
            if n <= len(cum): mint_w[str(w)] = float(np.max(np.abs(cum[:n])))
        def pkcm(w):
            seg = disp.slice(p_utc, p_utc + w).data
            return float(np.max(np.abs(seg - np.mean(seg))) * 100.0) if len(seg) > 2 else 0.0
        return (eid, s, band, Rkm, snr, mint_w, pkcm(3), pkcm(10), float(p_utc), float(cft), float(ref - 10.0))
    except Exception:
        return None

meas = []; done = 0; t0w = time.time()
with ThreadPoolExecutor(max_workers=NWORK) as ex:
    futs = [ex.submit(work, t) for t in tasks]
    for f in as_completed(futs):
        done += 1
        r = f.result()
        if r: meas.append(r)
        if done % 200 == 0:
            print("  harvest %d/%d, %d meas, %.0fs" % (done, len(tasks), len(meas), time.time() - t0w), flush=True)
            if done % 2000 == 0:
                json.dump({"events": evmeta, "measurements": meas, "partial": done},
                          open(OUTDIR + "/tmd_measurements.json.partial", "w"))
json.dump({"events": evmeta, "measurements": meas, "classify": classify,
           "fields": "eid,sta,band,Rkm,snr10,mint_w,pd3,pd10,p_utc,cft,p_pred"},
          open(OUTDIR + "/tmd_measurements.json", "w"))
print("harvest done: %d measurements" % len(meas), flush=True)

# ---- score: locate (first-4 EDT) + magnitude (Step-E _classical_M) vs TMD truth ----
by_ev = {}
for m in meas: by_ev.setdefault(m[0], []).append(m)
def classical_for(row, Rkm):
    _, sta, band, _, snr, mint_w, pd3, pd10, p_utc, cft, p_pred = row
    mw10 = mint_w.get("10")
    cl = {"pd3": pd3, "pd10": pd10, "mint": mw10 if mw10 else 0.0, "mint_w": mint_w, "clipped": False}
    return R._classical_M(cl, Rkm, sta)
rows = []
for eid, ms in by_ev.items():
    ev = evmeta[eid]
    picks = sorted([m for m in ms if m[8] and m[9] > 0], key=lambda m: m[8])
    if len(picks) < 4:
        rows.append({"eid": eid, "M": ev["M"], "n": len(picks), "located": False, "why": "<4 picks"}); continue
    first4 = picks[:4]; t0 = first4[0][8]
    arr = [{"station": m[1], "relative_time": float(m[8] - t0)} for m in first4]
    loc = R.locate_earthquake(arr)
    if not (loc and loc["rms"] <= R.MAX_RMS_LOCATE):
        rows.append({"eid": eid, "M": ev["M"], "n": len(picks), "located": False,
                     "why": ("rms %.1f" % loc["rms"]) if loc else "grid-edge/none"}); continue
    epi_km = gps2dist_azimuth(ev["lat"], ev["lon"], loc["lat"], loc["lon"])[0] / 1000.0
    ot_err = (t0 + loc["origin_time"]) - float(UTCDateTime(ev["time"]))
    mag = None; mag_sta = None
    cal = [m for m in picks if m[1] in R.MAG_CALIBRATED] or picks
    for m in cal[:1] + picks[:1]:
        Rkm_t = gps2dist_azimuth(loc["lat"], loc["lon"], R.STATION_COORDS[m[1]]["lat"], R.STATION_COORDS[m[1]]["lon"])[0] / 1000.0
        cm = classical_for(m, Rkm_t)
        if cm.get("Mwp", 0) > 0: mag, mag_sta = cm["Mwp"], m[1]; break
    rows.append({"eid": eid, "M": ev["M"], "region": ev.get("region", ""), "n": len(picks), "located": True,
                 "epi_km": round(epi_km, 1), "ot_err": round(ot_err, 2), "rms": round(loc["rms"], 2),
                 "Mwp": round(mag, 2) if mag else None, "mag_sta": mag_sta,
                 "snr_mag": next((m[4] for m in cal[:1]), None)})
json.dump({"rows": rows, "classify": classify, "n_events": len(cat),
           "n_candidates": len(evmeta) - n_nofetch}, open(OUT, "w"))

# ---- report ----
loc_rows = [r for r in rows if r.get("located")]
print("\n================ TMD RETROSPECTIVE (today's Step-E code) ================")
print("events tested: %d | candidates(>=4 stn in range): %d | reached >=4 picks: %d | LOCATED: %d"
      % (len(cat), len(evmeta) - n_nofetch, sum(1 for r in rows if r.get("n", 0) >= 4), len(loc_rows)))
if loc_rows:
    ep = np.array([r["epi_km"] for r in loc_rows]); ot = np.array([r["ot_err"] for r in loc_rows])
    print("epicentre error vs TMD: median %.0f km | 90%% < %.0f km" % (np.median(ep), np.percentile(ep, 90)))
    print("origin-time error vs TMD: median %+.1f s | 90%% |err| < %.1f s" % (np.median(ot), np.percentile(np.abs(ot), 90)))
    mg = [(r["M"], r["Mwp"]) for r in loc_rows if r.get("Mwp")]
    if mg:
        M, W = map(np.array, zip(*mg)); dm = W - M
        print("magnitude  our MwP vs TMD-ML (TMD ML uncalibrated): bias %+.2f | MAE %.2f (n=%d)" % (np.mean(dm), np.mean(np.abs(dm)), len(dm)))
        for lo, hi in [(0.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 10.0)]:
            s = dm[(M >= lo) & (M < hi)]
            if len(s): print("  TMD M%.1f-%.1f: n=%4d  our MwP-(TMD ML) mean %+.2f  MAE %.2f" % (lo, hi, len(s), np.mean(s), np.mean(np.abs(s))))
print("saved -> %s" % OUT)
print("DONE")
