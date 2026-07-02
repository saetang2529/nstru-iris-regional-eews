#!/usr/bin/env python
# AUTO-VENV (2026-06-10): allow a plain `python3 -u regional_eews.py` from any shell —
# if started OUTSIDE a venv, re-exec with the eews venv python + the usual env (the process is REPLACED,
# so exactly one python runs). Import-as-module is unaffected (__main__-gated).
if __name__ == "__main__":
    import os as _os, sys as _sys
    if _sys.prefix == _sys.base_prefix:
        _venv = _os.path.expanduser("~/eews_venv/bin/python")
        if _os.path.exists(_venv):
            _os.environ.setdefault("PYTHONWARNINGS", "ignore")
            _os.environ.setdefault("LANG", "C.UTF-8")
            _os.execv(_venv, [_venv, "-u"] + _sys.argv)
"""
regional_eews.py  —  Single-command regional IRIS/GEOFON EEWS using E3WS deep models.

SAME method as the NSTRU per-station code (nstru20_complete_e3ws.py) and the geophone
RPi server (geophone_eews_alert_monitor_with_epicenter_telegram.py).  The ONLY difference
is the data source: instead of reading an ADXL355/geophone sensor via a local C daemon,
this reads regional broadband stations LIVE over SeedLink.

  Detect  : E3WS DEEP DET model (DET_SLRZ, XGBoost 3-class)  — trigger
  Pick    : STA/LTA on the vertical (NOT the E3WS PICK AI)    — P arrival
  Magnitude: 8 E3WS StackingXGB tp models (tp3..tp10), tp10 default, MEDIAN across stations
  Velocity->acceleration: per-station velocity PZ (2 zeros at origin) ->
            st_FV(pb_inst=True, pzfile=...)  (DET 1-7 Hz, MAG 1-45 Hz) — identical to nstru20
  Locate  : 4-station L-BFGS-B + Huber loss, depth FIXED 10 km, iasp91 (P/p/Pn)
  Alert   : 2 Telegram channels (FAST 4-station + MONITOR partial/station-status)

Run with the E3WS conda env (python 3.8, sklearn 1.1.1 — MATCHES the trained models;
base `python` has sklearn 1.0.2 and warns / risks invalid MAG predictions):
    E3WS=/path/to/anaconda3/envs/E3WS/bin/python   # YOUR E3WS conda env (set per machine; launch scripts read $E3WS_PY)
    $E3WS regional_eews.py                 # run live
    $E3WS regional_eews.py --test-models   # load models/PZ/coords and exit
    $E3WS regional_eews.py --probe 60      # connect SeedLink, ingest 60 s, report, exit
"""
import warnings; warnings.filterwarnings("ignore")   # silence the harmless matplotlib Axes3D warning (3D unused; 2D Agg plots work) + obspy startup noise
import os, sys, time, threading, datetime, math, argparse, traceback, subprocess, glob
from collections import deque, defaultdict
import numpy as np

# ----------------------------------------------------------------------------- paths
# PORTABLE: every path is derived from THIS file's own folder, so the whole project folder can be copied to any
# computer and run unchanged — no hardcoded home paths. (2026-06-08.) The only environment-specific
# thing left is the python interpreter itself (the E3WS conda env), referenced by the launch scripts, not here.
PROJECT = os.path.dirname(os.path.abspath(__file__))                        # the folder THIS script lives in
BACKUP  = os.path.dirname(PROJECT)                                          # parent (only the now-unused E3WS_MODEL_DIR uses it)
E3WS_MODEL_DIR   = BACKUP + "/NSTRU_VELOCITY_TEMPLATE_20260307_073028_UTC/E3WS/models"   # UNUSED (E3WS DET/MAG skipped)
PZ_DIR  = PROJECT + "/responses/pz"
DET_PERSTATION_DIR = PROJECT + "/models/DET_perstation"                     # UNUSED (DET skipped) — removable from a portable copy
STATIONXML = PROJECT + "/responses/IRIS_TM_GE_HH_response.xml"
sys.path.insert(0, PROJECT)
# functions/pb_utils_v16.py (E3WS feature lib) is ONLY used by the legacy E3WS DET / tp-MAG path (never called now).
# Look for it under PROJECT, then the parent, then the legacy dir — OPTIONAL so a portable folder needn't ship it.
for _fp in (PROJECT, BACKUP):
    if os.path.isdir(os.path.join(_fp, "functions")):
        sys.path.insert(0, _fp); break

import joblib
import requests
from obspy import Stream, Trace, UTCDateTime, read_inventory
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, trigger_onset
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from scipy.optimize import minimize

try:
    from functions.pb_utils_v16 import st_FV      # E3WS feature vector (140 feats) — only the unused E3WS path needs it
except Exception:
    st_FV = None                                  # portable: functions/ not shipped -> fine, st_FV is never called (STA/LTA+MwP)
import stations_config as CFG

# ----------------------------------------------------------------------------- constants (from config / nstru20)
SAMPLE_RATE          = 100.0
DET_THRESHOLD        = CFG.TRIGGER.get("det_threshold", 0.80)   # mean of last 3 P-probs
DET_THRESHOLD_BY     = {"NPW": 0.30}    # per-station DET threshold override: NPW under-fires on regional P and is
                                        # BIMODAL (fires 0.5-0.9 on close+large events, else ~0.0; noise=0.00 always).
                                        # Lowering 0.4->0.3 is safe but ~no recall gain (misses are P_prob≈0); retrain NPW.
NOISE_FILTER         = 0.99
CONSECUTIVE_REQUIRED = 3
DET_WINDOW_SEC       = 12
PICK_WINDOW_SEC      = 30                                       # 10 s LTA warmup + 20 s scan
FIXED_DEPTH          = CFG.EVENT["fixed_depth_km"]              # 10 km
MIN_STATIONS         = CFG.EVENT["min_stations"]               # 4
ASSOC_WINDOW         = CFG.EVENT["assoc_window_sec"]           # 120 s
PARTIAL_TIMEOUT      = CFG.TELEGRAM["partial_timeout_sec"]     # 60 s
MAG_TP_DEFAULT       = CFG.EVENT.get("mag_tp_default", 10)
P_ASSOC_TOL          = 5.0
HUBER_DELTA          = 0.15
VP_AVERAGE           = 5.95
EVENT_COOLDOWN       = 30                                       # per-station re-trigger lockout (Step C-4 2026-06-11: was 90 — a foreshock/glitch trigger blinded the station to a mainshock 30-80 s later, 0/10 re-triggered in sim)
MAG_LAG              = 15                                       # extra wait for tp10 (P+13) + latency
BUFFER_SEC           = 150
N_MAG_MODELS         = 8                                        # tp3..tp10
# --- false-alarm guards (added 2026-06-06 after the overnight reconnect-burst test) ---
MIN_SNR_LOCATE   = CFG.TRIGGER.get("min_snr_locate", 0.0)      # SNR gate REMOVED: real regional P ~= noise in amplitude
                                                              # (median real-pick SNR 1.2) -> amplitude can't discriminate.
MAX_RMS_LOCATE   = CFG.TRIGGER.get("max_rms_locate", 4.0)      # PRIMARY gate: picks must fit ONE hypocentre (timing coherence)
EDT_SIGMA        = CFG.TRIGGER.get("edt_sigma", 1.0)          # EDT pick+model std (s) for the location likelihood
# --- multiscale EDT search (PREPARED 2026-06-08, DEFAULT OFF — no behaviour change until deployed) ---
# The single 0.3-deg coarse grid MISSES the sharp sigma=1s likelihood peak for ~half of station geometries
# -> locks onto a 3-of-6-pairs secondary maximum -> mislocates 130-250 km -> RMS>4 -> a REAL event is wrongly
# rejected. Fix: scan the coarse grid with a SMOOTHED sigma (broad peak the 0.3-deg step can't skip -> always
# the right basin), then refine locally with the true EDT_SIGMA -> pins the real peak. Same latency (~388 ms).
# Validated: synthetic 0-2 km, 455-event catalog +42% real-event alerts, RMS 6.9->1.6 s, epi 249->75 km.
# Deploy = set multiscale_locate=True (+ the sweep-chosen multiscale_sigma_coarse) in CFG.TRIGGER and restart.
MULTISCALE_LOCATE       = CFG.TRIGGER.get("multiscale_locate", False)
MULTISCALE_SIGMA_COARSE = CFG.TRIGGER.get("multiscale_sigma_coarse", 6.0)   # coarse-pass smoothing sigma (s)
# --- no-ML recursive STA/LTA trigger (Z-only; replaces the E3WS DET model — no training, add stations freely) ---
USE_STALTA_TRIGGER = CFG.TRIGGER.get("use_stalta_trigger", True)   # True = STA/LTA trigger; False = legacy E3WS DET
STALTA_STA   = CFG.TRIGGER.get("stalta_sta", 1.0)     # recursive STA window (s) — regional emergent P
STALTA_LTA   = CFG.TRIGGER.get("stalta_lta", 20.0)    # recursive LTA window (s)
STALTA_ON    = CFG.TRIGGER.get("stalta_on", 4.5)      # trigger-on ratio (tuned: 8/8 recall on M5.2, false 2.7/hr vs 5.5 at 3.5)
STALTA_OFF   = CFG.TRIGGER.get("stalta_off", 1.5)     # trigger-off ratio
STALTA_WIN   = CFG.TRIGGER.get("stalta_win", 40.0)    # detection window (s); must exceed LTA+STA+fresh
STALTA_FRESH = CFG.TRIGGER.get("stalta_fresh", 3.0)   # only trigger if P energy is in the last N s (= arriving now)
MIN_TRIG_DUR = CFG.TRIGGER.get("min_trig_dur", 1.0)   # sustained-duration quality gate (s) — rejects spikes
DET_BAND     = (CFG.TRIGGER.get("det_fmin", 2.0), CFG.TRIGGER.get("det_fmax", 8.0))  # detection bandpass (Hz)
WARMUP_SEC       = CFG.TRIGGER.get("warmup_sec", 120)          # suppress DET this long after a stream (re)start
STREAM_GAP_SEC   = 10.0                                        # sample-time gap larger than this = data discontinuity
STREAM_STALE_SEC = 20.0                                        # wall-clock pause larger than this = (re)connect
GRID_LAT         = (-11.0, 30.0)                               # locator search bounds (a solution AT the edge = unconstrained)
GRID_LON         = (88.0, 141.0)                               # Step D-1 (2026-06-11): widened to the 48-station network
                                                               # footprint (-10.5..29.7N, 91..140.7E). The old 5-26N/91-107E
                                                               # frame (22-station era) grid-edge-rejected every Sumatra-
                                                               # south-of-5.4N / Banda / Mindanao event — including the REAL
                                                               # M7.8 of 2026-06-07 (detected at 11 stations, never located).
                                                               # Public output still gated by RELEVANCE_KM / FAR_MAG.
GRID_EDGE_MARGIN = 0.4
BOT        = CFG.TELEGRAM["bot_token"]
CH_FAST    = CFG.TELEGRAM["CH_FAST"]
CH_MONITOR = CFG.TELEGRAM["CH_MONITOR"]
# NOT-FELT reroute (2026-06-22): a located event whose PUBLIC message carries the green
# "no effect in Thailand" line (i.e. province_effects/Mercalli finds NO felt province AND M<FAR_MAG) is
# sent to this OPERATOR channel ONLY instead of the public CH_FAST — anti-cry-wolf (don't annoy Thai
# people with quakes they cannot feel) while the operator still sees every located event. FELT events and
# every M>=FAR_MAG / tsunamigenic event ALWAYS stay on CH_FAST (the green line is never emitted for them).
CH_NOTFELT = CFG.TELEGRAM.get("CH_NOTFELT", 0)                # no-effect channel (set in stations_config)

# Permanent file logging (2026-06-10): every log() line ALSO appended to a monthly file
# eews_<YYYY-MM>.log in this folder; at each month roll-over the previous month is xz -9e compressed (HIGHEST, fully
# reversible — read later with `unxz file.xz` or `xz -d file.xz`) into log_archive/ and a fresh file is started.
# Terminal output is unchanged. Self-contained: works no matter how the EEWS is launched (no redirect/cron needed).
_LOG_LOCK = threading.Lock()
_LOGST = {"month": None, "fh": None, "path": None, "swept": False}

def _log_xz(path):
    """Background: xz -9e (max compression, reversible) a finished monthly log -> log_archive/; drop it if empty."""
    try:
        if not (path and os.path.exists(path)):
            return
        if os.path.getsize(path) == 0:
            os.remove(path); return
        subprocess.run(["xz", "-9e", "-f", path], check=False, timeout=1800)   # -9e=highest; `unxz`/`xz -d` to read later
        arch = os.path.join(PROJECT, "log_archive"); os.makedirs(arch, exist_ok=True)
        xzf = path + ".xz"
        if os.path.exists(xzf):
            os.replace(xzf, os.path.join(arch, os.path.basename(xzf)))
    except Exception:
        pass

def log(msg):
    line = "[%s] %s" % (datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)                                              # unchanged: still to the terminal
    try:
        mon = datetime.datetime.utcnow().strftime("%Y-%m")
        with _LOG_LOCK:
            if not _LOGST["swept"]:                                      # startup: compress any stray previous-month logs
                _LOGST["swept"] = True
                for p in glob.glob(os.path.join(PROJECT, "eews_*.log")):
                    if os.path.basename(p) != ("eews_%s.log" % mon):
                        threading.Thread(target=_log_xz, args=(p,), daemon=True).start()
            if _LOGST["month"] != mon:                                   # month rolled over -> rotate + compress old
                if _LOGST["fh"] is not None:
                    try: _LOGST["fh"].close()
                    except Exception: pass
                    threading.Thread(target=_log_xz, args=(_LOGST["path"],), daemon=True).start()
                _LOGST["month"] = mon
                _LOGST["path"] = os.path.join(PROJECT, "eews_%s.log" % mon)
                _LOGST["fh"] = open(_LOGST["path"], "a", buffering=1)    # line-buffered -> tail-able live
            _LOGST["fh"].write(line + "\n")
    except Exception:
        pass

# ----------------------------------------------------------------------------- MAG unpickle class (verbatim nstru20)
class pb_SAM():
    """Stacking model class used by E3WS MAG models (required for unpickling)."""
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

# ----------------------------------------------------------------------------- station coords + per-station PZ
HARDCODED_COORDS = {
    "NPW": (19.7785, 96.1376), "CMMT": (18.8140, 98.9444), "MHIT": (19.3148, 97.9632),
    "CMAI": (19.9325, 99.0453), "CRAI": (20.2289, 100.3734), "LOEI": (17.5093, 101.6244),
    "NAYO": (14.3152, 101.3209), "NONG": (18.0635, 103.1458), "PBKT": (16.5735, 100.9688),
    "PHRA": (18.4989, 100.2290), "PRAC": (12.4726, 99.7929), "SKLT": (7.1758, 100.6157),
    "SRDT": (14.3948, 99.1213), "SRIT": (8.5955, 99.6020), "TMDB": (13.6684, 100.6070),
    "UBPT": (15.2773, 105.4695),
    "TGI": (20.77, 97.03), "NGU": (21.21, 94.92),   # MM Myanmar source-side (western azimuth) 2026-06-06
    "LHMI": (5.23, 96.95), "GSI": (1.30, 97.58),     # GE Sumatra (southern azimuth)
    "LSA": (29.70, 91.13),                            # IC Lhasa/Tibet (NW azimuth)
    "KUM": (5.29, 100.65), "IPM": (4.48, 101.03),    # MY Malaysia (southern azimuth)
}
STATION_COORDS = {}     # {sta: {'lat':..,'lon':..}}
STATION_NET    = {}     # {sta: 'GE'/'TM'}
PZFILE         = {}     # {sta: path}
DISABLED_STATIONS = {"NONG"}   # only NONG stays off (all components dead + clock-skew).
# 2026-06-06: re-enabled ALL TM + NPW for AZIMUTHAL COVERAGE. Validation: full network gives location
# 279 km (vs 439 km on 8 stations) and recall 86% (vs 52%). False alarms are rejected by the RMS<=4
# coherence gate + warm-up guard, NOT by disabling stations. NPW (GE, Myanmar) stays enabled with a low
# DET threshold (DET_THRESHOLD_BY) for the critical western azimuth on Myanmar events.

def build_station_tables():
    """Coordinates from the saved StationXML (fallback to hardcoded); PZ path per station."""
    coords = {}
    try:
        inv = read_inventory(STATIONXML)
        for net in inv:
            for st in net:
                coords[st.code] = (st.latitude, st.longitude)
                STATION_NET[st.code] = net.code
    except Exception as e:
        log(f"  (inventory read failed: {e}; using hardcoded coords)")
    for src in CFG.SOURCES.values():
        for sta in src["stations"]:
            if sta in DISABLED_STATIONS:
                continue
            la, lo = coords.get(sta, HARDCODED_COORDS.get(sta))
            STATION_COORDS[sta] = {"lat": la, "lon": lo}
            STATION_NET.setdefault(sta, src["network"])
            PZFILE[sta] = f"{PZ_DIR}/{STATION_NET[sta]}_{sta}_HH.pz"

# ----------------------------------------------------------------------------- E3WS models
# DET trigger model assignment (2026-06-05): use the LANTA-trained PER-STATION model
# for EVERY station. The generic DET_SLRZ OVER-TRIGGERS on these broadband stations (verified:
# generic P-prob ~1.0 on quiet data vs ~0.0 for the per-station models) — that is exactly the
# false-alarm problem the per-station training fixes. Problem stations borrow the nearest
# same-sensor (CMG-3T 120 s) neighbour's model. Generic kept only as a last-resort fallback.
DET_ASSIGN = {
    "CMMT": "CMMT", "MHIT": "MHIT", "CRAI": "CRAI", "PHRA": "PHRA", "SKLT": "SKLT", "SRIT": "SRIT",  # Aug-24 own (140-feat, verified trigger)
    "NPW":  "NPW",    # own Aug-24 (140-feat); DET uses a 2-zero velocity PZ (DET_PZ) to match its training
    "UBPT": "UBPT",   # own Aug-03 (140-feat, 2-class, works)
    # The Aug-04 _fixed models are 141-FEATURE -> INCOMPATIBLE with pb_utils_v16 (140 feats) -> substitute
    # the nearest Aug-24 (140-feat) model. (Recommend retraining LOEI/NAYO/PBKT/PRAC/SRDT at 140 feats.)
    "LOEI": "PHRA",   # ~170 km
    "PBKT": "PHRA",   # ~220 km
    "NAYO": "PHRA",   # central, no close Aug-24
    "PRAC": "SRIT",   # ~430 km
    "SRDT": "SRIT",   # ~640 km, no close Aug-24
    "CMAI": "CMMT",   # ~125 km same CMG-3T
    "TMDB": "PHRA",   # offline; placeholder until it returns
    # NONG removed (all 3 components dead + clock-skew) -> see DISABLED_STATIONS
}
DET_GENERIC = None
DET_BY_STATION = {}
MAG_MODELS = []

def load_models():
    global DET_GENERIC, DET_BY_STATION, MAG_MODELS
    # Detection = recursive STA/LTA (no ML); magnitude = MwP@10s (Tsuboi). NO ML models are loaded — the old
    # per-station detection models and the tp-magnitude models have NO call sites (run_det / calculate_magnitude
    # are never called), so loading them only wasted ~1.4 GB RAM + startup time and tied the folder to an external
    # 245 MB model dir. Not loading them -> less RAM, faster start, self-contained / portable folder. (2026-06-08)
    DET_GENERIC, DET_BY_STATION, MAG_MODELS = None, {}, []
    log("  detection = STA/LTA · magnitude = MwP@10s")

# ----------------------------------------------------------------------------- E3WS feature / DET / pick / mag
def _three_comp(st):
    """Return a Z,N,E stream at 100 Hz, common window, EQUAL contiguous length
    (st_FV builds np.array([Z,N,E]) and needs identical npts; guard masked/short windows)."""
    try:
        comps = {}
        for comp in ("Z", "N", "E"):
            sel = st.select(component=comp)
            if not sel:
                return None
            tr = sel.merge(method=1, fill_value=0)[0]
            if hasattr(tr.data, "mask"):
                tr.data = tr.data.filled(0.0)
            if abs(tr.stats.sampling_rate - SAMPLE_RATE) > 0.01:
                tr.resample(SAMPLE_RATE)
            comps[comp] = tr
        t1 = max(tr.stats.starttime for tr in comps.values())
        t2 = min(tr.stats.endtime for tr in comps.values())
        if t2 - t1 < 5:
            return None
        out = Stream()
        for comp in ("Z", "N", "E"):
            tr = comps[comp].copy().trim(t1, t2, pad=True, fill_value=0)
            if hasattr(tr.data, "mask"):
                tr.data = tr.data.filled(0.0)
            out += tr
        n = min(len(tr.data) for tr in out)
        if n < 200:
            return None
        for tr in out:                               # contiguous float64, identical length -> st_FV safe
            tr.data = np.ascontiguousarray(tr.data[:n], dtype=np.float64)
        return out
    except Exception:
        return None

def run_det(stream, pzfile, model):
    """E3WS DET — returns (p_prob, noise_prob)."""
    feats = st_FV(stream, pb_inst=True, pzfile=pzfile, fmin=1.0, fmax=7.0)
    feats = np.nan_to_num(np.real(feats), nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)
    probs = model.predict_proba(feats)[0]          # [noise, P, S]
    return float(probs[1]), float(probs[0])

def pick_p_arrival(stream, fallback_time):
    """STA/LTA detect (2-10 Hz) then AIC refinement snaps the pick to the true first break — anchored on the
    STRONGEST CFT peak so a pre-event noise bump can't latch an early pick (Maeda 1985 / Kalkan 2016). The
    detection threshold (3.0) and the search window are UNCHANGED; ~1.7 ms. Returns (UTCDateTime, cft_strength).
    Validated 2026-06-08 on 600 USGS events: +46% detections, +26% locations, equal epi/origin-time, |dM| 0.25->0.22."""
    try:
        z = (stream.select(component="Z") or stream)[0].copy()   # Z if present, else the single best component
        z.filter("bandpass", freqmin=2.0, freqmax=10.0, zerophase=True, corners=4)
        df = z.stats.sampling_rate
        nsta, nlta = int(0.5 * df), int(10.0 * df)
        if len(z.data) < nlta + nsta:
            return fallback_time, 0.0
        cft = classic_sta_lta(z.data, nsta, nlta)
        scan = cft[nlta:]
        if scan.size == 0 or scan.max() < 3.0:                   # no real onset -> no pick (sensitivity unchanged)
            return fallback_time, 0.0
        i_anchor = nlta + int(np.argmax(scan))                   # strongest onset = the real P (robust to noise bumps)
        a = max(nlta, i_anchor - int(5.0 * df))                  # AIC refine: 5 s back from the peak
        b = min(len(z.data), i_anchor + int(0.5 * df))
        seg = z.data[a:b].astype(float)
        if len(seg) < int(1.0 * df):                             # too short -> fall back to the peak itself
            return z.stats.starttime + i_anchor / df, float(cft[i_anchor])
        n = len(seg); kk = np.arange(1, n)                       # Akaike Information Criterion onset (vectorised, O(n))
        c1 = np.cumsum(seg); c2 = np.cumsum(seg * seg)
        n1 = kk.astype(float); n2 = (n - kk).astype(float)
        v1 = (c2[kk - 1] - c1[kk - 1] ** 2 / n1) / n1
        v2 = ((c2[-1] - c2[kk - 1]) - (c1[-1] - c1[kk - 1]) ** 2 / n2) / n2
        aic = np.full(n, np.inf); good = (v1 > 1e-30) & (v2 > 1e-30)
        aic[kk[good]] = n1[good] * np.log(v1[good]) + n2[good] * np.log(v2[good])
        k = int(np.nanargmin(aic[1:-1])) + 1
        return z.stats.starttime + (a + k) / df, float(cft[i_anchor])
    except Exception as e:
        log(f"  STA/LTA error: {e}")
        return fallback_time, 0.0

def calculate_magnitude(stream, p_arrival, sec, pzfile):
    """tp magnitude at `sec` after P (nstru20): window P-7 .. P+3+sec, st_FV 1-45 Hz."""
    if sec < 3 or sec > 10:
        return 0.0
    try:
        st_mag = stream.copy().trim(p_arrival - 7, p_arrival + 3 + sec)
        if len(st_mag) < 3 or len(st_mag[0].data) < 100:
            return 0.0
        feats = np.nan_to_num(np.real(st_FV(st_mag, pb_inst=True, pzfile=pzfile, fmin=1.0, fmax=45.0)),
                              nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)
        idx = min(sec - 3, len(MAG_MODELS) - 1)
        return float(MAG_MODELS[idx].predict(feats)[0])
    except Exception as e:
        log(f"  MAG error: {e}")
        return 0.0

def snr_estimate(stream, p_arrival):
    """Rough SNR = rms(2 s after P) / rms(5 s before P) on vertical."""
    try:
        z = (stream.select(component="Z") or stream)[0]   # Z if present, else the single best component
        sig = z.slice(p_arrival, p_arrival + 2.0).data
        noi = z.slice(p_arrival - 5.0, p_arrival).data
        if len(sig) and len(noi) and np.std(noi) > 0:
            return float(np.std(sig) / np.std(noi))
    except Exception:
        pass
    return 0.0

# ----------------------------------------------------------------------------- location (verbatim geophone server)
TAUP = TauPyModel(model="iasp91")

def travel_time(slat, slon, sdep, stalat, stalon):
    d = locations2degrees(slat, slon, stalat, stalon)
    try:
        arr = TAUP.get_travel_times(source_depth_in_km=sdep, distance_in_degree=d, phase_list=["P", "p", "Pn"])
        if arr:
            return arr[0].time
    except Exception:
        pass
    km, _, _ = gps2dist_azimuth(slat, slon, stalat, stalon)
    return (km / 1000.0) / VP_AVERAGE

def _pred_abs_p(loc, t0, sta):
    """Predicted ABSOLUTE P time at sta for a located solution (epoch float)."""
    c = STATION_COORDS[sta]
    return t0 + loc["origin_time"] + float(np.interp(
        _dist_deg(loc["lat"], loc["lon"], c["lat"], c["lon"]), TT_DIST, TT_TIME))

def _loo_rescue(ordered):
    """Step D-2 (2026-06-11 MC study, v2 spec): ONE leave-one-out rescue round over the FIRST 5 picks when the
    first-4 locate failed. A candidate must (a) locate with RMS <= 1.5 s (real rescues measured 0.12-1.30), and
    (b) prove the excluded pick is a TRUE outlier (|residual| >= max(6 s, 2x RMS)). The candidate is NOT alerted
    here — it is held PENDING until an independent later pick confirms it (see _rescue_confirmed); MC-measured
    false-assembly with the full stack: ~0.1-0.9/yr at current pick rates, before the unchanged downstream gates.
    Multi-outlier far events (Mindanao-class) fragment across association windows — out of scope (Step H)."""
    pool = ordered[:5]
    best = None
    for i in range(len(pool)):
        sub = [r for j, r in enumerate(pool) if j != i]
        if len(sub) < MIN_STATIONS:
            continue
        t0s = min(r["P_utc"] for r in sub)
        arr = [{"station": r["station"], "relative_time": float(r["P_utc"] - t0s)} for r in sub]
        cand = locate_earthquake(arr)
        if not cand or cand["rms"] > 1.5:
            continue                                   # bar tightened 2.0->1.5: the D-1 wide grid fits noise more
                                                       # easily (~3x candidates measured) and real rescues sit at
                                                       # 0.12-1.30 s, so 1.5 keeps full recall with margin restored
        exc = pool[i]
        if exc["station"] not in STATION_COORDS:
            continue
        resid = abs(exc["P_utc"] - _pred_abs_p(cand, t0s, exc["station"]))
        if resid < max(6.0, 2.0 * cand["rms"]):
            continue                                   # excluded pick is NOT clearly the outlier -> unsafe
        if best is None or cand["rms"] < best["loc"]["rms"]:
            best = {"loc": cand, "subset": sub, "t0": t0s, "excluded": exc["station"], "exc_resid": resid}
    return best

def _rescue_confirmed(resc, cl):
    """Step D-2 confirmation: a PENDING rescue alerts only when an INDEPENDENT later pick (distinct station,
    in the cluster, not in the rescue subset) lands within +-4 s of the frozen solution's predicted P time
    (tightened from 6 s: correlated same-burst noise confirms at ~0.10 within 6 s, ~halved at 4 s — verifier
    re-measurement). Non-confirming picks do NOT kill the rescue (they may be outliers); it expires at close."""
    used = {r["station"] for r in resc["subset"]} | {resc["excluded"]}
    for r in cl:
        if r["station"] in used or r["station"] not in STATION_COORDS:
            continue
        if abs(r["P_utc"] - _pred_abs_p(resc["loc"], resc["t0"], r["station"])) <= 4.0:
            return r["station"]
    return None

def huber(r, delta=HUBER_DELTA):
    a = abs(r)
    return 0.5 * r * r if a < delta else delta * (a - 0.5 * delta)

def huber_objective(p, arrivals, depth):
    lat, lon, ot = p
    s = 0.0
    for a in arrivals:
        c = STATION_COORDS[a["station"]]
        s += huber(a["relative_time"] - (ot + travel_time(lat, lon, depth, c["lat"], c["lon"])))
    return s

def residuals(p, arrivals, depth):
    lat, lon, ot = p
    return np.array([a["relative_time"] - (ot + travel_time(lat, lon, depth, STATION_COORDS[a["station"]]["lat"],
                                                             STATION_COORDS[a["station"]]["lon"])) for a in arrivals])

# Robust GRID-SEARCH locator. The old L-BFGS-B (gradient) got stuck at the station centroid for
# regional events (epicentre far from the array, TauP phase-jump gradient) — verified: RMS 9-22 s
# even with PERFECT input times. Grid search over the region finds the global minimum.
TT_DIST = np.arange(0.0, 63.01, 0.05)   # epicentral distance grid (deg) — to 63°: the D-1 widened grid puts the far
                                        # corner (-11N,141E) 62.8° from LSA; flat extrapolation there would corrupt the
                                        # EDT surface. Startup build ~doubles (one-off, ~40-90 s on the Pi).
TT_TIME = None                          # P travel time vs distance at FIXED_DEPTH (built lazily)
TT_S    = None                          # S travel time vs distance (Step E: predicted S-P selects the MwP window)
_TT_LOCK = threading.Lock()             # guard the lazy build against a hub/silent-station thread race

def _ensure_tt():
    global TT_TIME, TT_S, FAR_BMIN, FAR_BMAX, FAR_SIDX
    if TT_TIME is not None and TT_S is not None and FAR_BMIN is not None:
        return
    with _TT_LOCK:
        if TT_TIME is not None and TT_S is not None and FAR_BMIN is not None:   # double-check after acquiring the lock
            return
        stations = sorted(STATION_COORDS)
        # Step H: DISK CACHE for the whole table set. The build (P+S sweeps + 3 extra depth sweeps for the
        # moveout bounds) costs minutes on the Pi; every restart after the first loads in <1 s. The key pins
        # everything the tables depend on — any station-set/grid/depth change forces a clean rebuild.
        coords_sig = ";".join("%s:%.4f,%.4f" % (s, STATION_COORDS[s]["lat"], STATION_COORDS[s]["lon"])
                              for s in stations)         # v2: coordinates pinned too (review: a coord fix must
        ckey = "|".join([coords_sig, "%d:%.2f" % (len(TT_DIST), float(TT_DIST[-1])),    # invalidate the bounds)
                         "fd%.0f" % FIXED_DEPTH, "g%s%s" % (GRID_LAT, GRID_LON), "step0.75", "iasp91",
                         "P,Pn,Pg,p+S,Sn,Sg,s", "d" + ",".join(str(d) for d in FAR_DEPTHS), "hv2"])
        cpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tt_cache.npz")
        try:
            z = np.load(cpath, allow_pickle=False)
            if str(z["key"]) == ckey:
                TT_S = z["tt_s"]; FAR_BMIN = z["bmin"]; FAR_BMAX = z["bmax"]
                FAR_SIDX = {s: i for i, s in enumerate(stations)}
                TT_TIME = z["tt_p"]                      # last: TT_TIME non-None is the "ready" flag others poll
                return
        except Exception:
            pass
        def _build(phases, dep=FIXED_DEPTH):
            tt = []
            for d in TT_DIST:
                try:
                    arr = TAUP.get_travel_times(source_depth_in_km=dep, distance_in_degree=float(d),
                                                phase_list=phases)
                    tt.append(min(a.time for a in arr) if arr else np.nan)
                except Exception:
                    tt.append(np.nan)
            ttv = np.array(tt)
            nan = np.isnan(ttv)                          # linear-fill any gap so np.interp never propagates NaN
            if nan.any() and (~nan).any():
                ttv[nan] = np.interp(TT_DIST[nan], TT_DIST[~nan], ttv[~nan])
            return ttv
        tt_p = _build(["P", "Pn", "Pg", "p"])
        TT_S = _build(["S", "Sn", "Sg", "s"])            # Step E (2026-06-11): S-P window selection for close stations
        # Step H (2026-06-11): pairwise P-moveout ENVELOPE over the whole alert grid — for every station pair
        # (i,j), the min/max possible P(j)-P(i) over any source in the grid at depths 10/100/300/600 km.
        # The far-track associator admits a pick iff its differential time vs EVERY member is physically
        # explainable by SOME gridded source (+- FAR_SLACK). Research-validated 2026-06-11 (H4+H2 design).
        slat = np.array([STATION_COORDS[s]["lat"] for s in stations])
        slon = np.array([STATION_COORDS[s]["lon"] for s in stations])
        N = len(stations)
        las = np.arange(GRID_LAT[0], GRID_LAT[1] + 1e-9, 0.75)
        los = np.arange(GRID_LON[0], GRID_LON[1] + 1e-9, 0.75)
        LA, LO = np.meshgrid(las, los, indexing="ij")
        gla, glo = LA.ravel(), LO.ravel()
        D = np.empty((gla.size, N))
        for k in range(N):
            D[:, k] = _dist_deg(gla, glo, float(slat[k]), float(slon[k]))
        bmin = np.full((N, N), np.inf); bmax = np.full((N, N), -np.inf)
        for dep in FAR_DEPTHS:
            ttd = tt_p if dep == FIXED_DEPTH else _build(["P", "Pn", "Pg", "p"], dep)
            TTg = np.interp(D, TT_DIST, ttd)
            for c0 in range(0, TTg.shape[0], 400):       # chunked: keeps the (g x N x N) temp under ~8 MB
                t = TTg[c0:c0 + 400]
                dt = t[:, None, :] - t[:, :, None]       # dt[g,i,j] = P(j)-P(i) for source g
                np.minimum(bmin, dt.min(axis=0), out=bmin)
                np.maximum(bmax, dt.max(axis=0), out=bmax)
        FAR_BMIN, FAR_BMAX = bmin, bmax
        FAR_SIDX = {s: i for i, s in enumerate(stations)}
        try:
            np.savez(cpath + ".tmp.npz", key=np.array(ckey), tt_p=tt_p, tt_s=TT_S, bmin=bmin, bmax=bmax)
            zv = np.load(cpath + ".tmp.npz", allow_pickle=False)        # v2: verify before swap (atomic via
            assert str(zv["key"]) == ckey and zv["bmin"].shape == bmin.shape   # rename; a torn write must not
            os.replace(cpath + ".tmp.npz", cpath)                        # poison every future restart)
        except Exception as _e:
            log("tt-cache save failed (non-fatal, will rebuild next start): %s" % _e)
        TT_TIME = tt_p                                   # last: TT_TIME non-None is the "ready" flag others poll

def _dist_deg(lat, lon, slat, slon):    # great-circle distance (deg), vectorised over stations
    la1, lo1, la2, lo2 = map(np.radians, [lat, lon, slat, slon])
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return np.degrees(2*np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))

def locate_earthquake(arrivals, depth=FIXED_DEPTH):
    """EDT (Equal Differential Time) likelihood location — uses ONLY the P arrival times of the
    first MIN_STATIONS picks. Origin-time-free and outlier-robust (Lomax/NonLinLoc; PRESTo RTLoc):
    the sum sits OUTSIDE the exponential, so the location is the point satisfying the MOST station
    pairs, and a single bad pick falls to ~0 weight instead of biasing the fit (the Huber objective
    it replaces still partially fits an outlier). Global grid-search max of the EDT stack, then a
    local fine grid (the TauP table isn't smooth -> grid, not gradient)."""
    stas = [a for a in arrivals if a["station"] in STATION_COORDS]
    if len(stas) < MIN_STATIONS:
        return None
    _ensure_tt()
    obs = np.array([a["relative_time"] for a in stas])
    slat = np.array([STATION_COORDS[a["station"]]["lat"] for a in stas])
    slon = np.array([STATION_COORDS[a["station"]]["lon"] for a in stas])
    ia, ib = np.triu_indices(len(stas), k=1)          # all station pairs
    dT_obs = obs[ia] - obs[ib]                         # observed differential times (origin-time independent)
    s2 = 2.0 * (EDT_SIGMA ** 2)                        # sigma_a^2 + sigma_b^2 (equal sigma per pick)
    # multiscale: the COARSE pass uses a SMOOTHED sigma (broad peak the 0.3-deg grid can't skip) while the
    # refine + RMS keep the true sigma. With MULTISCALE_LOCATE False, s2_coarse == s2 -> legacy search exactly.
    s2_coarse = 2.0 * (MULTISCALE_SIGMA_COARSE ** 2) if MULTISCALE_LOCATE else s2
    def predict(lat, lon):
        return np.interp(_dist_deg(lat, lon, slat, slon), TT_DIST, TT_TIME)
    def edt(lat, lon, s2=s2):                          # EDT likelihood (sum OUTSIDE exp -> outlier-robust)
        pred = predict(lat, lon)
        r = dT_obs - (pred[ia] - pred[ib])
        return float(np.sum(np.exp(-(r * r) / s2)))
    # 1) COARSE grid search = global maximum of the EDT stack (robust; no centroid/gradient trap).
    #    MULTISCALE: evaluate with the smoothed s2_coarse so the 0.3-deg step lands in the TRUE basin
    #    instead of skipping the sharp peak and locking onto a 3-of-6-pairs secondary maximum.
    las = np.arange(GRID_LAT[0], GRID_LAT[1] + 0.01, 0.3)         # Step D-1: VECTORISED coarse pass — same grid
    los = np.arange(GRID_LON[0], GRID_LON[1] + 0.01, 0.3)         # points/formula as the old double loop; argmax-
    LA, LO = np.meshgrid(las, los, indexing="ij")                 # identical (final output bitwise-equal, 25/25 probe;
    PRED = np.stack([np.interp(_dist_deg(LA, LO, float(slat[k]), float(slon[k])), TT_DIST, TT_TIME)
                     for k in range(len(stas))])                  # so the 6.4x-bigger D-1 grid costs LESS than the
    RES = dT_obs[:, None, None] - (PRED[ia] - PRED[ib])           # old 5-26N loop did)
    Lgrid = np.exp(-(RES * RES) / s2_coarse).sum(axis=0)
    jla, jlo = np.unravel_index(int(np.argmax(Lgrid)), Lgrid.shape)
    la, lo = float(las[jla]), float(los[jlo])
    # 2) FINE local grid around the coarse optimum.
    for step in (0.1, 0.03):
        gb = (edt(la, lo), la, lo)
        for dla in np.arange(-0.3, 0.3001, step):
            for dlo in np.arange(-0.3, 0.3001, step):
                L = edt(la + dla, lo + dlo)
                if L > gb[0]:
                    gb = (L, la + dla, lo + dlo)
        _, la, lo = gb
    # origin time + absolute-time RMS at the EDT optimum (RMS feeds the MAX_RMS_LOCATE coherence gate)
    pred = predict(la, lo); ot = float(np.median(obs - pred))
    rms = float(np.sqrt(np.mean((obs - (ot + pred)) ** 2)))
    # REJECT grid-edge solutions: pinned to the search boundary = unconstrained (one-sided force-fit).
    if (la <= GRID_LAT[0] + GRID_EDGE_MARGIN or la >= GRID_LAT[1] - GRID_EDGE_MARGIN
            or lo <= GRID_LON[0] + GRID_EDGE_MARGIN or lo >= GRID_LON[1] - GRID_EDGE_MARGIN):
        log(f"  locate: rejected grid-edge solution {la:.2f},{lo:.2f} (RMS {rms:.1f}s) — unconstrained")
        return None
    az = np.sort(np.array([gps2dist_azimuth(la, lo, float(sa), float(so))[1] for sa, so in zip(slat, slon)]))
    az_gap = float(np.max(np.diff(np.concatenate([az, az[:1] + 360.0])))) if len(az) >= 2 else 360.0
    return {"lat": float(la), "lon": float(lo), "depth": depth, "origin_time": ot, "az_gap": az_gap,
            "rms": rms, "n": len(stas), "success": True}

# ----------------------------------------------------------------------------- Telegram
# ---- MULTI-METHOD distance-corrected magnitude (all windows <= P+10 s) ----
# E3WS-tp (trained <=200 km, saturates ~M6) for fast small-M, + Pd (Kuyuk&Allen) and Mwp (Tsuboi)
# with their PUBLISHED physical slopes + a PER-STATION offset (own-station calibration), which
# preserve large-M scaling. Fused by regime; big events flagged. Coeffs fit vs the USGS catalog.
import json as _json
try: MAG_CALIB = _json.load(open(PROJECT + "/mag_distance_calib.json"))     # E3WS tp: M=Mraw+a+b*log10(R)
except Exception: MAG_CALIB = {}
try: PSCAL = _json.load(open(PROJECT + "/perstation_mag_calib.json"))       # Pd3_KA per-station offsets
except Exception: PSCAL = {}
try: MWPCAL = _json.load(open(PROJECT + "/mwp_dist_sta_calib.json"))["Mwp_dist_sta"]   # Mwp: +a+b*log10(R)+station_corr
except Exception: MWPCAL = {"a": 0.73, "b": 0.0, "station_term": {}, "global_term": 0.0}
# mbP per-station calibration (2026-06-13): a FAST P-window body-wave magnitude anchored to USGS mb, for the SMALL
# tier (< ~M4) where MwP over-reads ~+0.8 and Mw is the wrong unit for a small quake. mbP = log10(pd10cm) +
# a*log10(Rkm) + b + station_term. Validated bias -0.0 / MAE 0.28 vs USGS mb (held-out generalises); on the TMD
# M3-4 band it corrects MwP +0.82 -> -0.07. Used ONLY in the combine_magnitude small-tier selector below.
try: MBPCAL = _json.load(open(PROJECT + "/mbp_calib.json"))
except Exception: MBPCAL = None
if MBPCAL is not None and not (isinstance(MBPCAL, dict) and {"a", "b"} <= set(MBPCAL.keys())):
    MBPCAL = None   # review (2026-06-13): a parseable-but-PARTIAL calib (missing a/b) must DISABLE mbP, never KeyError the alert
# Magnitude-eligible = stations with their OWN Mwp station_term (calibrated). Others (global-term:
# MY.KUM/IPM, GE.GSI, NPW, TM-global) are ALERT + LOCATE only. Mwp is taken from the single FASTEST
# (first-triggering = closest = highest-SNR) calibrated station for fast EEW (2026-06-06).
MAG_CALIBRATED = set(MWPCAL.get("station_term", {}).keys())
# --- full-Mw (Mwpd) recal + tsunami-watch (2026-06-09; option-3 deploy: both features ON, public) ---
try:
    _MWPDJ = _json.load(open(PROJECT + "/mwpd_perstation_calib_20260609.json"))
    MWPDCAL = _MWPDJ; MWPD_ELIGIBLE = set(_MWPDJ.get("mag_eligible", []))
except Exception:
    MWPDCAL = {"global_term": 0.0, "station_term": {}}; MWPD_ELIGIBLE = set()
ENABLE_FULLMW = True            # at ~T+5min recompute the network-median Mwpd; post an UPDATE if it grew >=0.25 vs MwP@10s
ENABLE_TSUNAMI_WATCH = True     # if full Mw>=7.5 + offshore + Thai-threat -> tsunami watch (de-tide DART) posting to CH_FAST
EEWS_PROVINCE_MAP = True        # province felt-intensity MAP after the alert: M<6.3 with the first alert (rapid mag) via event_media Hook 1; M>=6.3 with the full-Mw recompute (final mag) via fullmw_tsunami Hook 2. Additive + crash-isolated; drawn only when something is felt.
try:
    import fullmw_tsunami as FMT
except Exception:
    FMT = None
try: INV = read_inventory(STATIONXML)                                       # full responses for Pd/Mwp
except Exception: INV = None

def _ps_off(method, sta):
    c = PSCAL.get(method)
    return (c.get("per_station", {}).get(sta, c.get("global_offset", 0.0)) if c else 0.0)

def _apply_calib(mraw, tp, Rkm):
    c = MAG_CALIB.get(str(tp))
    return mraw + c["a"] + c["b"] * np.log10(Rkm) if (c and mraw > 0 and Rkm > 0) else mraw

def _is_clipped(tr, min_run=5, amp_frac=0.98, flat_counts=2.0):
    """Flat-top / rail (clip) detector on RAW integer counts (2026-06-07). Returns True only if
    >= min_run consecutive samples are pinned near the extreme |amplitude| AND essentially IDENTICAL
    (|step| <= flat_counts counts) — the true digital/sensor saturation signature (a railed flat-top repeats
    the same value). A smooth large peak does NOT qualify (its samples still differ by tens-hundreds of
    counts), and a single spike fails (run < min_run). CONSERVATIVE: used only to PROTECT a real event (keep
    a railed station from vetoing it / being the magnitude station) and to flag its magnitude as a LOWER
    BOUND — it never creates an alert, so it cannot add false alarms."""
    try:
        d = tr.data.astype(float)
    except Exception:
        return False
    n = len(d)
    if n < min_run + 2:
        return False
    A = float(np.max(np.abs(d)))
    if A <= 0:
        return False
    near = np.abs(d) >= amp_frac * A                     # samples within 2% of the extreme (the rail)
    run = best = 0
    for i in range(n - 1):
        if near[i] and abs(d[i + 1] - d[i]) <= flat_counts:   # railed = consecutive samples ~identical (quantization)
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best >= min_run - 1                           # best counts transitions; min_run-1 transitions = min_run pinned samples (>=5 @100 Hz = >=50 ms) = clipped

def classical_measure(sta, p_arrival):
    """Distance-INDEPENDENT physical measurements from the raw vertical: Pd (cm) + moment-integral."""
    if INV is None:
        return {}
    try:
        with LOCK[sta]:
            st = BUFFERS[sta].slice(p_arrival - 8, p_arrival + 12).copy()
        z = st.select(component="Z")
        if not z:
            return {}
        tr = z.merge(method=1, fill_value=0)[0]
        clipped = _is_clipped(tr)                       # flat-top/rail check on RAW counts (before resample/deconv)
        # GAP-REJECT (2026-06-15): a real-time data gap is zero-filled by the merge above; the 0.01 Hz deconvolution
        # then turns the step into a low-freq swing that BALLOONS the displacement integral (a 0.3 s gap -> +0.6 mag;
        # this drove the live MHIT 4.83->5.46). Detect a run of consecutive EXACT-zeros in the P..P+10 window (real raw
        # counts are never exactly 0 that long) -> flag {"gap"} -> excluded from MAGNITUDE (still fine for LOCATION).
        # Post-merge / no get_gaps dependency, so robust to how BUFFERS is populated.
        _gw = tr.slice(p_arrival, p_arrival + 10).data
        if len(_gw) > 10:
            _gz = (np.asarray(_gw) == 0); _grun = 0; _gmax = 0
            for _gb in _gz:
                _grun = _grun + 1 if _gb else 0
                if _grun > _gmax: _gmax = _grun
            if _gmax > int(0.2 * tr.stats.sampling_rate):   # >0.2 s of exact zeros in the P window = a zero-filled gap
                return {"gap": True, "clipped": clipped}
        if abs(tr.stats.sampling_rate - SAMPLE_RATE) > 0.01:
            tr.resample(SAMPLE_RATE)
        d = tr.remove_response(inventory=INV, output="DISP", pre_filt=(0.01, 0.02, 40, 45))
        def pkcm(w):
            s = d.slice(p_arrival, p_arrival + w).data
            return float(np.max(np.abs(s - np.mean(s))) * 100.0) if len(s) > 2 else 0.0
        s = d.slice(p_arrival, p_arrival + 10).data
        mint = 0.0; mint_w = {}
        if len(s) > 10:
            cum = np.abs(np.cumsum(s - np.mean(s)) * d.stats.delta)
            mint = float(np.max(cum))
            for w in range(4, 11):                       # Step E: integral capped at every window 4..10 s — the
                n = int(w / d.stats.delta)               # S-P-capped window is SELECTED later, once the epicentre
                if n <= len(cum):                        # (hence predicted S-P) is known. Costs one cumsum (already
                    mint_w[w] = float(np.max(cum[:n]))   # computed) read at 7 points = free.
        pdw = {w: pkcm(w) for w in range(1, 11)}     # windowed PEAK displacement (cm) 1..10 s — for the mbP ts-tp cap
        return {"pd3": pkcm(3), "pd10": pkcm(10), "mint": mint, "mint_w": mint_w, "clipped": clipped, "pdw": pdw}
    except Exception:
        return {}

def _classical_M(cl, Rkm, sta):
    out = {}
    if cl.get("pd3", 0) > 0 and Rkm > 0:
        out["Pd"] = 1.23 * np.log10(cl["pd3"]) + 1.38 * np.log10(Rkm) + 5.39 + _ps_off("Pd3_KA", sta)
    if cl.get("mint", 0) > 0 and Rkm > 0:
        def _mwp_from(mint_val):
            Mo = mint_val * 4 * np.pi * 2600 * (6000.0 ** 3) * (Rkm * 1000.0)
            if Mo <= 0:
                return None
            base = (2.0 / 3.0) * (np.log10(Mo) - 9.1) + 0.2            # Tsuboi Mwp
            stc = MWPCAL["station_term"].get(sta, MWPCAL["global_term"])   # Kissling-style station correction
            return base + MWPCAL["a"] + MWPCAL["b"] * np.log10(Rkm) + stc
        m10 = _mwp_from(cl["mint"])
        out["Mwp"] = m10
        # Step E (2026-06-11, research consensus + 2 verifiers): at close stations the 10-s P window CONTAINS
        # the S wave (S-P < 11 s below ~90 km), inflating Mwp ~+0.3 for small/moderate events. Select the window
        # from the PREDICTED S-P at the located distance (iasp91 tables; existing a/b/station terms verified to
        # transfer: w=4 bias -0.05, w=5 -0.01). POLICY: below M6 report the capped value (kills the S over-read);
        # at/above M6 report max(capped, full) — never under-warn a large close event (capped under-reads ~0.3
        # for M7+; the T+5min full-Mw update then refines).
        if m10 is not None and cl.get("mint_w") and TT_S is not None:
            ddeg = Rkm / 111.19
            sp = float(np.interp(ddeg, TT_DIST, TT_S) - np.interp(ddeg, TT_DIST, TT_TIME))
            if sp < 11.0:
                w = int(max(4, min(10, sp - 1.0)))
                mw_int = cl["mint_w"].get(w, cl["mint_w"].get(str(w)))
                mcap = _mwp_from(mw_int) if mw_int else None
                if mcap is not None:
                    chosen = mcap if max(mcap, m10) < 6.0 else max(mcap, m10)
                    out["Mwp_w"] = mcap; out["Mwp_win"] = w; out["Mwp_sp"] = round(sp, 1)
                    out["Mwp_10"] = round(m10, 2)
                    out["Mwp"] = chosen
    return out

# Public saturation caveat thresholds — MwP UNDER-reads large events (measured onset ~M6.5):
MAG_CAVEAT_MIN = 6.3   # reported M >= this (or a clipped mag station) -> show the "may be larger" caveat
MAG_GREAT_MIN  = 7.0   # reported M >= this -> stronger "major earthquake" wording + the official-source note
# For LARGE events our rapid magnitude (and any single-network rapid magnitude) UNDER-reads (clipping) — so direct
# the public to the official government authority (TMD) for the authoritative magnitude. (2026-06-07.)
OFFICIAL_NOTE = ("\n📢 เหตุการณ์ใหญ่ — โปรดยึดข้อมูลทางการจากกรมอุตุนิยมวิทยา (TMD) earthquake.tmd.go.th"
                 "\n    Large event — please rely on the official source (Thai Met Dept / USGS) for the authoritative magnitude.")

# --- TSUNAMI-POTENTIAL flag (research, NOT an official warning) — added 2026-06-08 ---
# When the final magnitude is large AND the epicentre is OFFSHORE, append a bilingual note FLAGGING tsunami
# POTENTIAL and directing the public to the official Thai authority (NDWC / DDPM 1860). We flag potential only,
# never issue an evacuation order. Thresholds from PTWC / INCOIS / Thai-NDWC practice: M>=7.5 = tsunami-capable
# (PTWC auto-watch, NDWC evacuation-notify; no offshore M7.0-7.4 has damaged the Thai Andaman coast); M>=7.8 =
# regional/ocean-wide (NDWC's own alert trigger). NO depth gate: the locator fixes depth at 10 km (always shallow)
# and over-flagging is the safe direction for a "possible — follow NDWC" note.
TSUNAMI_FLAG_ON      = CFG.EVENT.get("tsunami_flag", True)
TSUNAMI_FLAG_MW      = CFG.EVENT.get("tsunami_flag_mw", 7.5)
TSUNAMI_OCEANWIDE_MW = CFG.EVENT.get("tsunami_oceanwide_mw", 7.8)
try:
    from global_land_mask import globe as _GLOBE
except Exception:
    _GLOBE = None
def _is_offshore(lat, lon):
    """True if (lat,lon) is in the sea (global_land_mask, ~1 km, ~0.4 ms). If the mask is unavailable -> False
    (fail safe: never raise a tsunami flag on an unverifiable location -> no false tsunami alarm on land)."""
    if _GLOBE is None:
        return False
    try:
        return not bool(_GLOBE.is_land(float(lat), float(lon)))
    except Exception:
        return False
def tsunami_note(M, lat, lon):
    """Bilingual TSUNAMI-POTENTIAL note (or '') — fires for an OFFSHORE great event. SATURATION-AWARE (2026-06-14):
    MwP@10s hard-saturates ~7 so a true M7.8 shows ~7.0; the old M>=TSUNAMI_FLAG_MW(7.5) gate suppressed this note
    even when the watch armed. Now gated at MAG_GREAT_MIN(7.0)=TSUNAMI_MW_FAST (consistent with the instant arm)."""
    if not TSUNAMI_FLAG_ON or M < MAG_GREAT_MIN or not _is_offshore(lat, lon):
        return ""
    note = ("\n🌊 เฝ้าระวังสึนามิ (อัตโนมัติ ไม่เป็นทางการ): แผ่นดินไหวใหญ่ใต้ทะเล อาจเกิดสึนามิ — "
            "โปรดติดตามและทำตามประกาศ ศูนย์เตือนภัยพิบัติแห่งชาติ (ปภ.) โทร 1860 ซึ่งจะยืนยันหรือยกเลิก"
            "\n    🌊 Tsunami watch (auto, unofficial): large undersea quake — tsunami possible. Follow Thailand's "
            "NDWC (DDPM 1860 · police 191), which will confirm or cancel.")
    if M >= TSUNAMI_OCEANWIDE_MW:                       # NDWC's own public-alert trigger -> regional / ocean-wide
        note += ("\n    ระดับภูมิภาค — ชายฝั่งอันดามัน (ภูเก็ต พังงา กระบี่ ระนอง ตรัง สตูล) อาจได้รับผลกระทบ"
                 "\n    Regional — Andaman coast (Phuket, Phang Nga, Krabi, Ranong, Trang, Satun) may be affected.")
    return note

def _robust_median(v):
    """Network-magnitude aggregation: median, but first DEMOTE a single gross over-reading station (per-station
    magnitude > median + 1.0) — ONLY when the magnitude is sub-great (median < 6.0). ASYMMETRIC (high-only) so a
    legitimate LOW close-station reading is never dropped; SCOPED to < 6.0 so a potential great event is NEVER
    under-called (it can only LOWER -> cannot create a false alarm). Validated 2026-06-20: 0/972 events with
    median>=6.3 changed, 0-worse on M4.5-7.5, fixes the SURA-class gross outlier (3.22->2.88). deployed."""
    m = float(np.median(v))
    if m >= 6.0:
        return m                                    # never touch potential great events (bigflag/FAR_MAG/tsunami)
    keep = [x for x in v if x <= m + 1.0]           # drop only impossible-HIGH over-readers (e.g. SURA's 50x glitch)
    return float(np.median(keep)) if keep else m

def combine_magnitude(reps, loc):
    """SINGLE-METHOD magnitude = Mwp (Tsuboi P-moment, 10 s of P) + per-station offset, median over
    the closest stations. NO regime fusion: a real incoming event's magnitude is unknown, so we use
    one method that scales across M4.5-~8 (best calibration here, MAE 0.20, slope ~1.1). Pd and
    E3WS-tp are computed only as CROSS-CHECKS (not used for the reported magnitude).
    Returns (M_final, label, big_flag, detail)."""
    mwps, pds, e3, mbps = [], [], [], []
    for r in reps:
        c = STATION_COORDS.get(r["station"])
        R = gps2dist_azimuth(loc["lat"], loc["lon"], c["lat"], c["lon"])[0] / 1000.0 if c else 0.0
        cm = _classical_M(r.get("cl", {}), R, r["station"])
        if "Mwp_win" in cm:    # S-P capping engaged at this magnitude station (close event)
            log("  Mwp window capped to %ds (S-P %.1fs) at %s: capped %.2f vs full10 %.2f -> using %.2f" % (
                cm["Mwp_win"], cm["Mwp_sp"], r["station"], cm["Mwp_w"], cm.get("Mwp_10", 0.0), cm.get("Mwp", 0.0)))
        if cm.get("Mwp", 0) > 0: mwps.append(cm["Mwp"])
        if cm.get("Pd", 0) > 0: pds.append(cm["Pd"])
        # mbP (small-event tier): fast P-window body-wave magnitude, USGS-mb-anchored. Skip a clipped/railed station.
        # ts-tp CAP (2026-06-13): measure the peak displacement over a P-ONLY window min(10 s, S-P-1) [floor MBPCAL.floor],
        # mirroring MwP's Step-E, so S-wave energy at close stations cannot inflate the "P" magnitude. capcorr re-centers
        # the P-only window against the fixed-10 s calibration. Falls back to pd10 (fixed 10 s) if pdw / TT-table absent.
        _cl = r.get("cl", {}) or {}
        _pdw = _cl.get("pdw") or {}; _pd10 = _cl.get("pd10", 0.0)
        if MBPCAL and R > 0 and not _cl.get("clipped") and (_pdw or (_pd10 and _pd10 > 0)):
            _capc = 0.0
            if _pdw and TT_S is not None:
                _dd = R / 111.19
                _sp = float(np.interp(_dd, TT_DIST, TT_S) - np.interp(_dd, TT_DIST, TT_TIME))
                _w = int(np.clip(np.floor(_sp - 1.0), int(MBPCAL.get("floor", 2)), 10))
                _pdm = _pdw.get(_w) or _pdw.get(str(_w)) or _pd10
                if _w < 10: _capc = MBPCAL.get("capcorr", 0.0)
            else:
                _pdm = _pd10                         # no pdw / TT table -> fixed-10 s fallback (no capcorr)
            if _pdm and _pdm > 0:
                mbps.append(np.log10(_pdm) + MBPCAL["a"] * np.log10(R) + MBPCAL["b"] + _capc
                            + MBPCAL.get("station_term", {}).get(r["station"], 0.0))
        mraw = r["mags"].get(MAG_TP_DEFAULT) or (max(r["mags"].values()) if r["mags"] else 0.0)
        me = _apply_calib(mraw, MAG_TP_DEFAULT, R)
        if me > 0: e3.append(me)
    M_mwp = _robust_median(mwps) if mwps else 0.0
    M_pd = float(np.median(pds)) if pds else 0.0
    M_e3 = float(np.median(e3)) if e3 else 0.0
    M_mbp = _robust_median(mbps) if mbps else 0.0
    if M_mwp > 0:
        M, label = M_mwp, "Mwp@10s"
    elif M_pd > 0:
        M, label = M_pd, "Pd (Mwp unavailable)"
    else:
        M, label = M_e3, "E3WS (fallback)"
    # SMALL-EVENT TIER (2026-06-13): below ~M4.5, MwP over-reads and Mw is the wrong unit for a small quake, so a real
    # M3 would headline ~M3.8 (public fear / cry-wolf). When the de-biased estimate is small, report the USGS-mb-anchored
    # mbP instead. Boundary M4.5 chosen by a 14,852-event test vs USGS mb (the ONLY magnitude truth here): selector bias
    # +0.004 at M4.5 vs +0.103 at M4.0 (TMD-ML is NOT truth -> used for decision only, never as the magnitude yardstick).
    # Gate on min(mbP, MwP) so the MwP over-read cannot keep a true small event in the MwP tier. mbP REPLACES only the
    # small tier; mid (MwP) + large (Mwpd via bigflag) stay. mbP is reported in mb units (~0.17 below Mw); set "mb_to_mw"
    # in mbp_calib.json to add a constant shift if one continuous Mw scale is preferred.
    _mbp_shift = (MBPCAL.get("mb_to_mw", 0.0) if MBPCAL else 0.0)
    # SMALL-TIER GATE (2026-06-18, refined to mbP-alone < 4.05; SUPERSEDES the 2026-06-13 BOTH-<4.5 review-BLOCKER-1 gate).
    # The BOTH gate left genuine small FELT events stuck in the MwP tier: a true M3-4 reads MwP at its ~M4.4-4.6 noise
    # floor (>=4.5), so the old "M_mwp<4.5" was False and the cleaner mbP never fired -> the public saw the inflated MwP.
    # Now switch on mbP alone. Fine-grid backtest (16,431 USGS + 960 TMD events, faithful combine_magnitude replay,
    # Lenovo mbp_gate_grid.py): X=4.05 corrects 93 small felt events with ZERO downgrade of any real M>=4.5 (first
    # real-moderate harm only at X>=4.10; bootstrap harm45 = 0 [0..0]; robust across >=3/4/5-station cutoffs). SAFE despite
    # dropping the MwP<4.5 guard because the magnitude is the network MEDIAN over up to 8 stations and a real M>=5 reads
    # mbP >= ~5 (mb-anchored), so the median mbP is never <4.05 -> a real M6+ can never be downgraded into this tier.
    if M_mwp > 0 and M_mbp > 0 and M_mbp < 4.05:
        M, label = M_mbp + _mbp_shift, "mbP@P (small, USGS-mb)"
    # Mwp UNDER-reads (saturates) for large events: measured onset ~M6.5 (bias -0.37 @6.5-7, -0.75 >M7).
    # Public "may be larger" caveat from reported M >= MAG_CAVEAT_MIN (set below the 6.5 onset because the
    # number is itself under-read -> a true M6.7 may report ~6.3) OR if the magnitude station clipped/railed.
    clipped_mag = any(r.get("cl", {}).get("clipped") for r in reps)   # railed magnitude station -> reported M is a lower bound
    big = (M >= MAG_CAVEAT_MIN) or clipped_mag
    detail = {"Mwp": round(M_mwp, 2), "Pd_xchk": round(M_pd, 2), "E3WS_xchk": round(M_e3, 2), "mbP": round(M_mbp, 2)}
    return M, label, big, detail

# ----------------------------------------------------------------------------- Telegram
# All sends are SERIALIZED + rate-limit-aware. The 4 parallel per-station plot threads each call these, so a
# raw burst of ~16 messages (8 photos + 8 docs) trips Telegram's 429 limit and the later ones get DROPPED
# (that was the "13 s from 4 stations but 6 min from only 2" bug). One lock + 429-retry (respect retry_after)
# + gentle pacing -> nothing is dropped; the fastest-ready plot still goes first, just queued one at a time.
_TG_LOCK = threading.Lock()

_TG_DRYRUN = os.environ.get("EEWS_DRY_RUN", "").strip().lower() not in ("", "0", "false", "no", "off")
# ^ hard kill-switch: tests/replays set EEWS_DRY_RUN=1 so a leaked daemon tick can NEVER reach the real channel
#   (review H2-5). Only an explicit truthy value enables it, so a stray EEWS_DRY_RUN=0 cannot silently mute live alerts.
def _tg_call(method, *, json=None, data=None, files_path=None, files_field=None):
    if _TG_DRYRUN:
        log("  [DRY-RUN] telegram %s suppressed (EEWS_DRY_RUN set)" % method)
        return None
    with _TG_LOCK:
        for _ in range(6):
            try:
                if files_path:
                    with open(files_path, "rb") as f:
                        r = requests.post(f"https://api.telegram.org/bot{BOT}/{method}",
                                          data=data, files={files_field: f}, timeout=120)
                else:
                    r = requests.post(f"https://api.telegram.org/bot{BOT}/{method}", json=json, timeout=15)
                if r.status_code == 429:                                   # rate-limited -> wait the told time, resend
                    w = 3
                    try: w = int(r.json()["parameters"]["retry_after"])
                    except Exception: pass
                    log("  telegram 429 -> wait %ss, retry" % w); time.sleep(min(w + 1, 30)); continue
                if r.status_code >= 500:                                   # transient server error -> retry
                    time.sleep(2); continue
                time.sleep(0.5)                                            # pacing so the next queued send stays under the limit
                return r
            except Exception as e:
                log("  telegram %s error: %s" % (method, e)); time.sleep(2)
        log("  telegram %s FAILED after retries" % method)
        return None

def tg(chat, text):
    _tg_call("sendMessage", json={"chat_id": chat, "text": text})

def tg_photo(chat, path, caption=""):
    _tg_call("sendPhoto", data={"chat_id": chat, "caption": caption}, files_path=path, files_field="photo")

def tg_doc(chat, path, caption=""):
    _tg_call("sendDocument", data={"chat_id": chat, "caption": caption}, files_path=path, files_field="document")

# ------------------------------------------------------- real-event media (bilingual report + 2 plots)
# Matches the ADXL355 RPi alerter, adapted for IRIS (2026-06-06):
#   - ONE station (the magnitude station): Z-only, instrument-removed VELOCITY (m/s) — not raw counts;
#     Y scientific x10^, X dual UTC(bottom)+ICT(top). Plot A=13 s (P-3..P+10); Plot B=6 min (P-60..P+300, ~5 min later).
#   - concise bilingual TH/EN text + Google Maps; time = Thai (ICT, Buddhist year) first, then UTC.
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import matplotlib.dates as _mdates
    from matplotlib.ticker import ScalarFormatter as _ScalarFormatter
    _PLOTS_OK = True
except Exception as _e:                                     # pragma: no cover
    _PLOTS_OK = False
    log(f"  matplotlib unavailable -> event plots disabled: {_e}")

try:
    _LOGO = _plt.imread(PROJECT + "/nstru_logo.png") if _PLOTS_OK else None   # NSTRU logo for the plot corner
except Exception:
    _LOGO = None

_UTC_TZ = datetime.timezone.utc
_ICT_TZ = datetime.timezone(datetime.timedelta(hours=7))
_GEO_UA = {"User-Agent": "NSTRU-IRIS-EEWS/1.0 (light2529@gmail.com)"}
_SEA = [((4, 16, 92, 98.5), ("ทะเลอันดามัน", "Andaman Sea")), ((4.5, 23, 106, 121), ("ทะเลจีนใต้", "South China Sea")),
        ((10, 22, 80, 92), ("อ่าวเบงกอล", "Bay of Bengal")), ((-10, 4, 90, 104), ("มหาสมุทรอินเดีย", "Indian Ocean"))]
TH_MON = {1: "ม.ค.", 2: "ก.พ.", 3: "มี.ค.", 4: "เม.ย.", 5: "พ.ค.", 6: "มิ.ย.", 7: "ก.ค.", 8: "ส.ค.", 9: "ก.ย.", 10: "ต.ค.", 11: "พ.ย.", 12: "ธ.ค."}
_STA_LOC_OVERRIDE = {"TMDB": "10"}   # Step C-6 (2026-06-11): TM.TMDB HHZ exists ONLY at loc '10' (epoch 2021-02-25->);
                                     # the TM-wide loc '00' matched nothing -> TMDB had NO data at all (dead roster slot)
_STA_SRC = {}    # sta -> (net, loc, Zchannel, is_geofon) from config (static)
for _src in CFG.SOURCES.values():
    _b = _src["channels"][0][:2]; _isg = "geofon" in _src["seedlink"]
    for _s in _src["stations"]:
        _STA_SRC[_s] = (_src["network"], _STA_LOC_OVERRIDE.get(_s, _src["location"]), _b + "Z", _isg)
_FDSN = {}

def _fdsn(is_geofon):
    key = "GEOFON" if is_geofon else "IRIS"
    if key not in _FDSN:
        from obspy.clients.fdsn import Client
        _FDSN[key] = Client(key, timeout=60)
    return _FDSN[key]

def severity(m):
    if m >= 8: return "มหากาพย์", "Epic"
    if m >= 7: return "ใหญ่มาก", "Great"
    if m >= 6: return "ใหญ่", "Major"
    if m >= 5: return "รุนแรง", "Strong"
    if m >= 4: return "ปานกลาง", "Moderate"
    if m >= 3: return "เบา", "Light"
    return "เล็ก", "Minor"

def _rev_geo(lat, lon, lang):
    r = requests.get("https://nominatim.openstreetmap.org/reverse",
        params={"lat": lat, "lon": lon, "format": "jsonv2", "accept-language": lang, "zoom": 14, "addressdetails": 1},
        headers=_GEO_UA, timeout=8)
    return r.json().get("address", {})

def _levels(a):
    sub = a.get("suburb") or a.get("quarter") or a.get("village") or a.get("municipality") or a.get("town") or ""
    dis = a.get("county") or a.get("city_district") or a.get("city") or a.get("district") or ""
    pro = a.get("state") or a.get("province") or a.get("region") or ""
    return sub, dis, pro, a.get("country") or ""

def _is_th(s):                             # True iff the string actually contains Thai script (else Nominatim gave local script, e.g. Burmese)
    return any("฀" <= ch <= "๿" for ch in (s or ""))

def geocode(lat, lon):
    """Detailed bilingual location -> (thai, english): subdistrict/district/province/country where available;
    sea-region or coords fallback. Best-effort with short timeout (never blocks the alert for long)."""
    try:
        ath = _rev_geo(lat, lon, "th"); time.sleep(1.1); aen = _rev_geo(lat, lon, "en")
    except Exception:
        ath = aen = {}
    sth, sen = _levels(ath), _levels(aen)
    if not sth[3] and not sen[3]:
        for (a, b, c, d), (th, en) in _SEA:
            if a <= lat <= b and c <= lon <= d: return th, en
        return ("%.2f°N %.2f°E" % (lat, lon),) * 2
    if ("ไทย" in (sth[3] or "")) or ("Thailand" in (sen[3] or "")):
        pt = []
        if sth[0]: pt.append("ต." + sth[0].replace("ตำบล", "").strip())
        if sth[1]: pt.append("อ." + sth[1].replace("อำเภอ", "").strip())
        if sth[2]: pt.append("จ." + sth[2].replace("จังหวัด", "").strip())
        th = " ".join(pt) if pt else (sth[3] or "ประเทศไทย")
    else:                                  # FOREIGN: full chain like English; Thai name per level where Nominatim has
        lv = []                            # one, else the English name (never raw Burmese script); no จังหวัด/ประเทศ dup
        for i in range(3):                 # subdistrict, district, province
            t = (sth[i] or "").replace("จังหวัด", "").strip(); e = (sen[i] or "").strip()
            v = t if _is_th(t) else e
            if v: lv.append(v)
        ct = (sth[3] or "").replace("ประเทศ", "").strip()
        ct = ct if _is_th(ct) else (sen[3] or "")
        if ct: lv.append(ct)
        th = ", ".join([x for x in lv if x]) or sth[3]
    return th, ", ".join([x for x in sen if x])

def _prefilt(sr):
    return (0.01, 0.02, 40.0, 45.0) if sr >= 80 else (0.01, 0.02, sr * 0.40, sr * 0.45)

def _to_vel(tr):
    t = tr.copy(); t.detrend("demean"); t.detrend("linear"); t.taper(0.05)
    if abs(t.stats.sampling_rate - SAMPLE_RATE) > 0.01: t.resample(SAMPLE_RATE)
    t.remove_response(inventory=INV, output="VEL", pre_filt=_prefilt(t.stats.sampling_rate), water_level=60)
    return t

# matplotlib's pyplot has process-global state and is NOT thread-safe -> serialize ONLY the figure build+save
# across the 4 parallel per-station plot threads (the slow FDSN fetch + Telegram upload stay fully parallel).
_MPL_LOCK = threading.Lock()

def _plot_event(vel, p_time, mag, net, sta, loc, kind, before, after, path):
    t = vel.copy().trim(p_time - before, p_time + after)
    if len(t.data) < 5:
        return None
    x = t.times("matplotlib")
    fig, ax = _plt.subplots(figsize=(11.5, 4.4))
    pk_amp = float(np.max(np.abs(t.data))) if len(t.data) else 1.0
    exp = int(np.floor(np.log10(pk_amp))) if pk_amp > 0 else -6
    sc = 10.0 ** exp
    ax.plot(x, t.data / sc, color="navy", lw=0.8)
    _pp = (loc.get("sp_p") or {}).get(sta)               # PhaseNet-refined P actually used in the location (2-of-2 consensus); Decision B 2026-06-16
    if _pp is not None:                                  # show the PhaseNet pick — the STA/LTA pick is only the trigger; the located/origin P is this one
        _pu = UTCDateTime(_pp)
        ax.axvline(_mdates.date2num(_pu.datetime), color="red", ls="--", lw=1.7, label="P pick (PhaseNet)")
    else:                                                # no consensus on this station -> the located P fell back to STA/LTA; show that
        ax.axvline(_mdates.date2num(p_time.datetime), color="red", ls="--", lw=1.6, label="P arrival (STA/LTA)")
    _st = (loc.get("sp_s") or {}).get(sta)               # consensus S-pick used in the location refinement (if any)
    if _st is not None:                                  # green marker, drawn ONLY when the S falls inside this window
        _su = UTCDateTime(_st)                           # (so 6-min shows every used S; 13-s shows it only when S-P <= 10 s)
        if p_time - before <= _su <= p_time + after:
            ax.axvline(_mdates.date2num(_su.datetime), color="green", ls="-.", lw=1.7, label="S pick (used)")
    ax.set_ylabel("Velocity Z  (×10$^{%d}$ m/s)" % exp)   # exponent in the label avoids the top-left ×10 offset clashing with the ICT axis
    ax.xaxis_date(); ax.xaxis.set_major_locator(_mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(_mdates.DateFormatter("%H:%M:%S", tz=_UTC_TZ)); ax.set_xlabel("UTC")
    ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(ax.get_xticks())
    ax2.xaxis.set_major_formatter(_mdates.DateFormatter("%H:%M:%S", tz=_ICT_TZ)); ax2.set_xlabel("ICT (UTC+7)")
    ax.set_title("%s.%s · Z · M%.1f · %s\nvelocity m/s, instrument-removed · epi %.2f,%.2f  d%.0fkm"
                 % (net, sta, mag, kind, loc["lat"], loc["lon"], loc["depth"]))
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    # NSTRU copyright (2026-06-06): faint diagonal watermark (hard to crop) + bottom credit,
    # so a reused plot still carries the attribution. (Provide an NSTRU logo PNG to also embed the logo.)
    fig.text(0.5, 0.5, "NSTRU IRIS EEWS", ha="center", va="center", fontsize=42, color="gray", alpha=0.07, rotation=28, zorder=0)
    fig.text(0.995, 0.006, "© NSTRU — IRIS Regional EEWS · research (do not reuse without attribution)",
             ha="right", va="bottom", fontsize=6.5, color="0.45")   # English only: matplotlib font has no Thai glyphs
    if _LOGO is not None:
        try:
            bb = ax.get_position()                   # real axes box (figure coords, after tight_layout)
            lw, lh = 0.062, 0.135                     # logo box size (figure fraction); imshow preserves aspect
            lx = bb.x0 + 0.036                        # balanced left margin (same height); stays left of the P arrival on both windows
            ly = bb.y1 - lh - 0.040                   # a bit BELOW the top frame -> never touches the ICT time ticks
            lax = fig.add_axes([lx, ly, lw, lh], anchor="NW", zorder=10); lax.imshow(_LOGO); lax.axis("off")
        except Exception:
            pass
    try:
        fig.savefig(path, dpi=300, format="jpg")
    finally:
        _plt.close(fig)                                  # always free the figure, even if savefig raises (no leak)
    return path

# Two-stage magnitude (2026-06-07): at ~T+5min recompute a FULLER Mw from a LONGER P window
# (extended-window Mwpd) over the calibrated stations whose S-P is large enough to extend (far stations) ->
# recovers MwP@10s's high-end under-read for big events (validated on M7.7 Sagaing: ~7.1 -> ~7.4). Used ONLY to
# push an UPDATE for a likely-saturated event (gated on bigflag) -> it can never create an alert.
MWPD_MAX_WIN = 40.0      # s — longest P-integration window
MWPD_MIN_WIN = 18.0      # s — a station must extend to >= this (S-P big enough) to add value beyond MwP@10s
MWP_UPDATE_DELTA = 0.25  # send an UPDATE only if the fuller Mw grew by >= this (catches M7.7 Sagaing 7.1->7.37; above the 12-stn median's jitter)

# Station -> short bilingual place name for the alert provenance line (built by build_placenames.py reverse-geocode).
try:
    _PLACENAMES = _json.load(open(os.path.join(PROJECT, "station_placenames.json"), encoding="utf-8"))
except Exception:
    _PLACENAMES = {}
def _station_place(sta):
    """(th, en) short place name for a station (province/city), falling back to the station code."""
    p = _PLACENAMES.get(sta) or {}
    th = (p.get("th") or "").replace("อำเภอเมือง", "").replace("อำเภอ", "").replace("จังหวัด", "").replace("รัฐ", "").strip() or sta
    en = (p.get("en") or "").replace("Mueang ", "").replace(" District", "").replace(" Tehsil", "").replace("City of ", "").replace("Kota ", "").strip() or sta
    return th, en

def recompute_mw_5min(loc, o_utc):
    """Extended-window Mwpd over the FAR calibrated stations (median). Returns (mw, n_sta, max_win, used_stations)."""
    if INV is None:
        return None, 0, 0.0, []
    dep = min((loc.get("depth", 10) or 10), 60)
    mws = []; maxwin = 0.0; used = []
    for sta in MAG_CALIBRATED:
        c = STATION_COORDS.get(sta)
        if not c:
            continue
        try:
            Rkm = gps2dist_azimuth(loc["lat"], loc["lon"], c["lat"], c["lon"])[0] / 1000.0
            d = locations2degrees(loc["lat"], loc["lon"], c["lat"], c["lon"])
            arrP = TAUP.get_travel_times(source_depth_in_km=dep, distance_in_degree=d, phase_list=["P", "Pn", "Pg", "p"])
            arrS = TAUP.get_travel_times(source_depth_in_km=dep, distance_in_degree=d, phase_list=["S", "Sn", "Sg", "s"])
            if not arrP or not arrS:
                continue
            Pg = o_utc + min(a.time for a in arrP)
            smp = min(a.time for a in arrS) - min(a.time for a in arrP)
            win = min(MWPD_MAX_WIN, 0.9 * smp)
            if win < MWPD_MIN_WIN:                          # too close to extend past the S wave -> MwP@10s already covers it
                continue
            net, locc, chz, is_g = _STA_SRC.get(sta, (STATION_NET.get(sta, ""), "", "HHZ", False))
            st = _fdsn(is_g).get_waveforms(net, sta, locc if locc else "*", chz, Pg - 15, Pg + win + 15)
            zz = st.merge(method=1, fill_value=0).select(component="Z") if st else None
            if not zz:
                continue
            tr = zz[0]
            if abs(tr.stats.sampling_rate - SAMPLE_RATE) > 0.01:
                tr.resample(SAMPLE_RATE)
            trf = tr.copy().detrend("demean").filter("bandpass", freqmin=2, freqmax=8, corners=4, zerophase=True)
            sig = trf.slice(Pg, Pg + 10).data; noi = trf.slice(Pg - 10, Pg).data
            if len(sig) < 10 or len(noi) < 10:
                continue
            if np.sqrt(np.mean(sig ** 2)) / (np.sqrt(np.mean(noi ** 2)) + 1e-12) < 5:   # SNR>5 quality gate
                continue
            nyq = 0.5 * tr.stats.sampling_rate
            pf = (0.01, 0.02, min(40, 0.8 * nyq), min(45, 0.9 * nyq))
            dd = tr.slice(Pg - 8, Pg + win + 10).remove_response(inventory=INV, output="DISP", pre_filt=pf)
            s = dd.slice(Pg, Pg + win).data
            if len(s) <= 10:
                continue
            mint = float(np.max(np.abs(np.cumsum(s - np.mean(s)) * dd.stats.delta)))
            mw = _classical_M({"mint": mint}, Rkm, sta).get("Mwp", 0)
            if mw > 0:
                mws.append(mw); maxwin = max(maxwin, win); used.append(sta)
        except Exception:
            continue
    if not mws:
        return None, 0, 0.0, []
    return float(np.median(mws)), len(mws), maxwin, used

def _fmt_delay(sec):
    """Alert delay as a friendly bilingual string: seconds under 1 min, else minutes+seconds (then hours+minutes)."""
    sec = int(round(max(0.0, sec)))
    if sec < 60:
        return "%d วินาที" % sec, "%d s" % sec
    m, s = divmod(sec, 60)
    if m < 60:
        return (("%d นาที %d วินาที" % (m, s)) if s else ("%d นาที" % m),
                ("%d min %d s" % (m, s)) if s else ("%d min" % m))
    h, m = divmod(m, 60)
    return "%d ชม. %d นาที" % (h, m), "%d h %d min" % (h, m)

def _time_footer(o_utc):
    """Shared 3-line bilingual footer appended to every SECONDARY public (CH_FAST) post so all of them show
    origin time (ICT+UTC), elapsed-since-origin, and post wall-clock (ICT+UTC) — exactly like event_media().
    Each line is prefixed with '\n' so it concatenates onto an existing message string. Robust to o_utc being
    a float epoch OR a UTCDateTime (wrapped). _post is captured ONCE so the post-time and the elapsed agree."""
    o = UTCDateTime(o_utc); o_ict = o + 7 * 3600
    ict_th = "%d %s %d  %s น." % (o_ict.day, TH_MON[o_ict.month], o_ict.year + 543, o_ict.strftime("%H:%M:%S"))
    utc_s = o.strftime("%Y-%m-%d %H:%M:%S")
    _post = UTCDateTime()
    dly_th, dly_en = _fmt_delay(float(_post - o))
    post_ict = (_post + 7 * 3600).strftime("%H:%M:%S")
    post_utc = _post.strftime("%Y-%m-%d %H:%M:%S")
    return ("\n🕑 %s (เวลาไทย)  ·  %s UTC"
            "\n⏱️ %s หลังเกิด / %s after the quake"
            "\n📤 โพสต์ %s น. (เวลาไทย)  ·  %s UTC") % (ict_th, utc_s, dly_th, dly_en, post_ict, post_utc)

def event_media(loc, Mfinal, magrep, origin, npicks, bigflag=False, locinfo=None, locstas=None, nmag=1):
    """Background thread: ONE bilingual alert (MwP@10s + station provenance + lower-bound caveat & official-TMD note for
    large events) + Plot A (13 s) immediately and Plot B (6 min) ~5 min later, both labeled MwP@10s. (Mwpd reverted: +0.4 bias.)"""
    try:
        sta = magrep["station"]
        net, locc, chz, is_g = _STA_SRC.get(sta, (STATION_NET.get(sta, ""), "", "HHZ", False))
        p_time = UTCDateTime(magrep["P_utc"])
        c = STATION_COORDS.get(sta, {})
        dist = gps2dist_azimuth(loc["lat"], loc["lon"], c["lat"], c["lon"])[0] / 1000.0 if c else 0.0
        th_loc, en_loc = geocode(loc["lat"], loc["lon"])
        o_utc = UTCDateTime(origin); o_ict = o_utc + 7 * 3600
        ict_th = "%d %s %d  %s น." % (o_ict.day, TH_MON[o_ict.month], o_ict.year + 543, o_ict.strftime("%H:%M:%S"))
        utc_s = o_utc.strftime("%Y-%m-%d %H:%M:%S")
        _post = UTCDateTime()                  # capture the post instant ONCE so post-time and delay agree (no double-clock skew)
        lat_s = float(_post - o_utc)           # SPEED: seconds from quake origin to this alert post (shown by design)
        dly_th, dly_en = _fmt_delay(lat_s)     # show as min+sec when >=1 min, not raw seconds (e.g. 190s -> 3 นาที 10 วินาที)
        post_ict = (_post + 7 * 3600).strftime("%H:%M:%S")   # bot post wall-clock in ICT/UTC+7 (same shift as o_ict on line 1170)
        post_utc = _post.strftime("%Y-%m-%d %H:%M:%S")       # bot post wall-clock in UTC (full date, unambiguous across midnight)
        # PUBLIC alert (2026-06-17): TWO MONOLINGUAL messages — all-Thai then all-English — each =
        # warning + per-province felt-effects footer merged, with the "research, not official" note as the LAST line.
        # Station/SNR + mag-station provenance + the one-sided ±100 km line REMOVED (by design). NO source-severity
        # word, NO method jargon (MwP/EDT/RMS). DO show the alert speed (origin->post). Plots unchanged below.
        mapu = "https://www.google.com/maps?q=%.4f,%.4f" % (loc["lat"], loc["lon"])
        TH = ["🚨 แผ่นดินไหว ขนาด %.1f" % Mfinal,
              "📍 %s" % th_loc,
              "🌍 พิกัด %.3f°N, %.3f°E · ลึก %.0f กม." % (loc["lat"], loc["lon"], loc["depth"]),
              "🕑 %s (เวลาไทย) · %s UTC" % (ict_th, utc_s),
              "⏱️ แจ้งเตือน %s หลังเกิด" % dly_th,
              "📤 โพสต์ %s น. (เวลาไทย) · %s UTC" % (post_ict, post_utc),
              "🗺️ แผนที่: %s" % mapu]
        EN = ["🚨 EARTHQUAKE  M%.1f" % Mfinal,
              "📍 %s" % en_loc,
              "🌍 %.3f°N, %.3f°E · depth %.0f km" % (loc["lat"], loc["lon"], loc["depth"]),
              "🕑 %s (Thailand) · %s UTC" % (o_ict.strftime("%H:%M:%S"), utc_s),
              "⏱️ alert %s after the quake" % dly_en,
              "📤 posted %s (Thailand) · %s UTC" % (post_ict, post_utc),
              "🗺️ map: %s" % mapu]
        # felt-in-Thailand? — the per-province footer is empty when nothing is felt; this drives the no-effect line + alarm gating
        _footer_failed = False                 # FAIL-SAFE (2026-07-02): footer FAILURE is not "genuinely empty"
        try:                                   # per-province felt-effects footer (guarded), merged into each language block
            if TT_S is not None:
                from province_effects import build_footer
                _fth, _fen = build_footer(Mfinal, float(o_utc), loc["lat"], loc["lon"], loc.get("depth", FIXED_DEPTH), TT_DIST, TT_S)
            else:
                _fth = _fen = ""; _footer_failed = True
        except Exception as _pe:
            _fth = _fen = ""; _footer_failed = True; log("province footer skipped: %s" % _pe)
        felt = bool(_fth or _fen)
        if not felt:                           # NO felt area in TH -> calm reassurance at LINE 2 (BLUF); felt/alarm lines suppressed (no contradiction)
            if Mfinal >= FAR_MAG:              # any BIG event (M>=6.5) the model can't place a province for can STILL sway BKK towers (MwP SATURATES: a true M7+ reads ~6.5-6.9) -> NEVER a flat "no effect"
                TH.insert(1, "🟡 แผ่นดินไหวใหญ่ระยะไกล — อาคารสูงในไทย (โดยเฉพาะกรุงเทพฯ) อาจโยกช้า ๆ ได้ หากรู้สึกโยกให้อยู่ในความสงบ หลบห่างหน้าต่าง และติดตามประกาศ TMD/ปภ.")
                EN.insert(1, "🟡 Great quake at long distance — tall buildings in Thailand (especially Bangkok) may sway slowly. If you feel swaying, stay calm, move away from windows, and follow TMD/NDWC.")
            elif _footer_failed:               # FAIL-SAFE: cannot claim "no effect" when the footer FAILED -> post
                # PUBLIC with a neutral line (no green line -> routes to CH_FAST below) + warn the operator channel.
                TH.insert(1, "ℹ️ ระบบประเมินผลกระทบขัดข้องชั่วคราว — โปรดติดตามประกาศ TMD")
                EN.insert(1, "ℹ️ Impact assessment temporarily unavailable — follow TMD.")
                try: tg(CH_MONITOR, "⚠️ province footer FAILED for M%.1f %s — alert routed PUBLIC without a no-effect claim; check province_effects/TT_S" % (Mfinal, en_loc))
                except Exception: pass
            else:
                _off = _is_offshore(loc["lat"], loc["lon"])
                TH.insert(1, "🟢 ไม่มีผลกระทบต่อประเทศไทย" + (" · ไม่มีภัยสึนามิ" if _off else ""))
                EN.insert(1, "🟢 No effect in Thailand." + (" · no tsunami threat" if _off else ""))
        if bigflag and felt:                   # magnitude caveat + Drop-Cover-Hold ONLY when Thailand is actually affected (never alongside a no-effect line)
            if Mfinal >= MAG_GREAT_MIN:        # probable great event -> stronger wording + official-source note
                TH += ["⚠️ แผ่นดินไหวขนาดใหญ่มาก — ขนาดที่แสดงเป็น 'ค่าโดยประมาณ' และเป็นค่าต่ำสุด เพราะวิธีคำนวณขนาดแบบเร็วจะอิ่มตัว (saturate) กับแผ่นดินไหวใหญ่มาก ขนาดจริงอาจสูงกว่านี้มาก คาดแรงสั่นรุนแรงและนานกว่าปกติ",
                       "📢 เหตุการณ์ใหญ่ — โปรดยึดข้อมูลทางการจากกรมอุตุนิยมวิทยา (TMD) earthquake.tmd.go.th"]
                EN += ["⚠️ Major earthquake — the magnitude shown is APPROXIMATE and a lower bound: the rapid magnitude method saturates for great quakes, so the true magnitude may be much larger. Expect strong, prolonged shaking.",
                       "📢 Large event — please rely on the official source (Thai Met Dept / USGS) for the authoritative magnitude."]
            else:                              # large / likely under-read (reported M >= 6.3 or clipped)
                TH += ["⚠️ ขนาดที่แสดงเป็น 'ค่าโดยประมาณ' และอาจต่ำกว่าจริง เพราะวิธีคำนวณขนาดแบบเร็ว (MwP) จะอิ่มตัว (saturate) กับแผ่นดินไหวใหญ่ — ขนาดจริงอาจสูงกว่านี้"]
                EN += ["⚠️ Magnitude shown is APPROXIMATE and may be under-estimated — the rapid magnitude method (MwP) saturates for large quakes, so the true magnitude may be higher."]
            TH += ["🟠 ผู้ที่อยู่ใกล้ศูนย์กลาง: หมอบ–ป้อง–ยึดเกาะ และระวังสิ่งของหล่น"]   # research: lead w/ ACTION (conditional on 'near')
            EN += ["🟠 Near the epicentre: Drop, Cover, Hold On; watch for falling objects."]
        if TSUNAMI_FLAG_ON and Mfinal >= MAG_GREAT_MIN and _is_offshore(loc["lat"], loc["lon"]):   # 🌊 tsunami-POTENTIAL (follow NDWC)
            TH += ["🌊 เฝ้าระวังสึนามิ (อัตโนมัติ ไม่เป็นทางการ): แผ่นดินไหวใหญ่ใต้ทะเล อาจเกิดสึนามิ — โปรดติดตามและทำตามประกาศ ศูนย์เตือนภัยพิบัติแห่งชาติ (ปภ.) โทร 1860 ซึ่งจะยืนยันหรือยกเลิก"]
            EN += ["🌊 Tsunami watch (auto, unofficial): large undersea quake — tsunami possible. Follow Thailand's NDWC (DDPM 1860 · police 191), which will confirm or cancel."]
            if Mfinal >= TSUNAMI_OCEANWIDE_MW:
                TH += ["ระดับภูมิภาค — ชายฝั่งอันดามัน (ภูเก็ต พังงา กระบี่ ระนอง ตรัง สตูล) อาจได้รับผลกระทบ"]
                EN += ["Regional — Andaman coast (Phuket, Phang Nga, Krabi, Ranong, Trang, Satun) may be affected."]
        thai = "\n".join(TH) + (("\n\n" + _fth) if _fth else "") + "\n\nℹ️ NSTRU EEWS · ระบบวิจัย ไม่ใช่ประกาศทางการ"
        eng  = "\n".join(EN) + (("\n\n" + _fen) if _fen else "") + "\n\nℹ️ NSTRU EEWS · research, not official"
        # NOT-FELT REROUTE (2026-06-22): if the green "no effect in Thailand" line is present
        # (province_effects/Mercalli found NO felt province AND M<FAR_MAG), post to the OPERATOR channel ONLY,
        # not the public channel — the text AND the 4-station plots below all follow out_chan. Felt events and
        # any M>=FAR_MAG/tsunami never carry the green line, so they always route to CH_FAST (public). Egress-only.
        out_chan = CH_NOTFELT if ("🟢 ไม่มีผลกระทบต่อประเทศไทย" in thai) else CH_FAST
        _route_tag = ""   # tag removed 2026-07-02: no-effect posts now go to the PUBLIC "Uneffect event" channel, so the operator-only wording is wrong there
        tg(out_chan, thai + "\n\n━━━━━━━━━━━━\n\n" + eng + _route_tag)   # ONE merged post: Thai + divider + English (same time)
        log("*** FAST ALERT (TH+EN merged%s) ***\n" % (" — OPERATOR-ONLY: no effect in Thailand" if out_chan != CH_FAST else "") + thai + "\n--- EN ---\n" + eng)
        if not _PLOTS_OK:
            return
        # PLOTS: the 4 LOCATING stations (origin-time + epicentre), each its OWN plot, SAME style as before.
        # 4 parallel threads per stage -> whichever station's plot finishes first is sent first (fastest).
        # Magnitude = network MEDIAN over the calibrated good-pick stations (2026-06-15); its value is the event M on every panel.
        # Mwpd REVERTED 2026-06-07: it over-reads ~+0.4 (fixed 40-s window over-integrates); MwP@10s is the only
        # validated estimator -> use it + the lower-bound note for big/clipped events.
        plot_stas = (locstas[:4] if locstas else [magrep])   # fallback: just the magnitude station (old behaviour)

        def _do_13s(r):                                    # 13 s (P-3..P+10) from the LIVE buffer (no network)
            s = r["station"]; pt = UTCDateTime(r["P_utc"])
            nt, lc, cz, ig = _STA_SRC.get(s, (STATION_NET.get(s, ""), "", "HHZ", False))
            try:
                trz = get_z_window(s, pt - 5, pt + 12)
                if not trz:
                    log("  plot(13s) %s.%s skipped — P %s window not in live buffer" % (nt, s, pt.strftime("%H:%M:%S")))
                    return
                with _MPL_LOCK:                            # serialize ONLY the draw (pyplot is not thread-safe)
                    p = _plot_event(_to_vel(trz[0]), pt, Mfinal, nt, s, loc, "13 s (P-3...P+10)", 3, 10, "/tmp/iris_evt_13s_%s.jpg" % s)
                if p:                                  # 300 dpi FILE only (no inline photo) — the downloadable file is preferred
                    tg_doc(out_chan, p, "%s.%s Z · velocity m/s · 13 s (P-3..P+10) · 300 dpi" % (nt, s))
            except Exception:
                log("  plot A (13 s) %s error:\n" % s + traceback.format_exc())

        def _do_6min(r):                                   # 6 min (P-1m..P+5m) via FDSN re-fetch ~5 min after THAT station's P
            s = r["station"]; pt = UTCDateTime(r["P_utc"])
            nt, lc, cz, ig = _STA_SRC.get(s, (STATION_NET.get(s, ""), "", "HHZ", False))
            try:
                wait = float((pt + 305) - UTCDateTime())   # each station self-times its own P+5min window
                if wait > 0:
                    time.sleep(min(wait, 400))
                st6 = _fdsn(ig).get_waveforms(nt, s, lc if lc else "*", cz, pt - 70, pt + 310)
                if not st6:
                    log("  plot(6min) %s.%s skipped — FDSN returned no data for P %s" % (nt, s, pt.strftime("%H:%M:%S")))
                    return
                with _MPL_LOCK:
                    p = _plot_event(_to_vel(st6.merge(method=1, fill_value=0)[0]), pt, Mfinal, nt, s, loc,
                                    "6 min (P-1m...P+5m)", 60, 300, "/tmp/iris_evt_6min_%s.jpg" % s)
                if p:                                  # 300 dpi FILE only (no inline photo)
                    tg_doc(out_chan, p, "%s.%s Z · velocity m/s · 6 min (P-1min..P+5min) · 300 dpi" % (nt, s))
            except Exception:
                log("  plot B (6 min) %s error:\n" % s + traceback.format_exc())

        for r in plot_stas:                                # 13 s — 4 parallel, fastest-first
            threading.Thread(target=_do_13s, args=(r,), daemon=True).start()
        for r in plot_stas:                                # 6 min — 4 parallel; each self-delays to its own P+5min
            threading.Thread(target=_do_6min, args=(r,), daemon=True).start()

        # PROVINCE FELT-MAP (Hook 1, additive + crash-isolated): a small/normal FELT event gets the map NOW with the
        # rapid magnitude, in parallel with the plots (shares _MPL_LOCK -> a "9th drawer"). BIG events (bigflag, M>=6.3)
        # SKIP this and get their map from the full-Mw recompute (Hook 2) so it uses the FINAL, not under-reading, magnitude.
        if EEWS_PROVINCE_MAP and felt and not bigflag and TT_S is not None:
            def _do_provmap():
                try:
                    import province_map
                    _out = "/tmp/iris_provmap_%d.jpg" % int(float(o_utc))
                    with _MPL_LOCK:                                # serialize ONLY the draw (pyplot not thread-safe)
                        _nf = province_map.render(loc, Mfinal, float(o_utc), loc.get("depth", FIXED_DEPTH),
                                                  TT_DIST, TT_S, _out, place_th=th_loc)
                    if _nf:
                        tg_doc(out_chan, _out, "🗺️ ความรุนแรงที่คาดว่าจะรู้สึกได้ รายจังหวัด (MMI) · M%.1f · 300 dpi" % Mfinal)
                except Exception:
                    log("  provmap (Hook1) error:\n" + traceback.format_exc())
            threading.Thread(target=_do_provmap, daemon=True).start()
    except Exception:
        log("  event_media error:\n" + traceback.format_exc())

# ----------------------------------------------------------------------------- live buffers (SeedLink ingest)
BUFFERS   = defaultdict(Stream)        # sta -> Stream(Z,N,E)
LAST_DATA = {}                         # sta -> UTCDateTime of newest sample
LOCK      = defaultdict(threading.Lock)
PKT_COUNT = defaultdict(int)
WARMUP_UNTIL  = {}                     # sta -> wall time until which DET is suppressed (response warm-up)
LAST_PKT_WALL = {}                     # sta -> wall time the last packet arrived (reconnect detection)
PKT_SPAN = defaultdict(lambda: 3.0)    # EMA of miniSEED record span (s): ~sub-3 s for 100-sps HH, 25-100 s for 20-sps BH (Step C-1)
for _s4, (_n4, _l4, _c4, _g4) in _STA_SRC.items():
    if _c4.startswith("BH"):
        PKT_SPAN[_s4] = 45.0               # seed slow 20-sps stations so they are not blind while the EMA learns after a restart
WARMUP_RESCAN = set()                  # stations owed ONE retro detection pass when their warm-up expires (Step C-3)
LAST_SCANNED = {}                      # sta -> buffer end (UTCDateTime) of the last det scan (skip-if-no-new-data, v2)
LAST_TRIG = {}                         # sta -> last trigger onset epoch (float) — same-onset re-trigger guard (v2)

def ingest(trace):
    sta = trace.stats.station
    if sta not in STATION_COORDS:
        return
    nowall = time.time()
    with LOCK[sta]:
        prev_end = LAST_DATA.get(sta)
        last_wall = LAST_PKT_WALL.get(sta)
        # WARM-UP GUARD (Step C-1 2026-06-11): arm ONLY on first data or a true SAMPLE-TIME discontinuity.
        # The old wall-clock (>20 s packet-spacing) re-arm kept every slow 20-sps BH station permanently in
        # warm-up (their records take 25-100 s to fill) — IPM/KUM/KKM/KSM/JAY logged ZERO triggers in 41 h
        # and the whole southern azimuth was detection-dead. A real outage still re-arms via the sample gap
        # its data hole leaves behind; sample-continuous-but-bursty delivery does NOT ring the deconvolution.
        if (prev_end is None
                or (trace.stats.starttime - prev_end) > STREAM_GAP_SEC):
            WARMUP_UNTIL[sta] = nowall + WARMUP_SEC
            WARMUP_RESCAN.add(sta)        # Step C-3: owed one retro-scan at expiry (a P during warm-up was lost forever)
            log(f"  {sta}: stream (re)start -> {WARMUP_SEC:.0f}s detection warm-up")
        LAST_PKT_WALL[sta] = nowall
        try:
            PKT_SPAN[sta] = min(150.0, 0.8 * PKT_SPAN[sta] + 0.2 * min(150.0, max(0.0, float(trace.stats.endtime - trace.stats.starttime))))
        except Exception:
            pass                                   # v3 (review B-4): clamped — one corrupt record header must not
                                                   # inflate the liveness allowance / worker deadline for hours
        BUFFERS[sta] += trace
        try:
            BUFFERS[sta].merge(method=-1)
        except Exception:
            BUFFERS[sta] = Stream([t for t in BUFFERS[sta]])
        try:
            end = max(t.stats.endtime for t in BUFFERS[sta])
            BUFFERS[sta].trim(end - BUFFER_SEC, end)
            LAST_DATA[sta] = end
        except Exception:
            pass
        PKT_COUNT[sta] += 1

# Per-station bad/dead components to replace with a good one (2026-06-05).
# Fill from the 5-min raw-count plot, e.g. {"PBKT": ["E"], "NONG": ["N", "E"]}. A listed (or
# auto-detected flatline, std < BAD_STD_MIN) component is overwritten by a good component (prefer Z).
BAD_COMPONENTS = {
    "CMAI": ["N"],        # CMAI North bad (std 417 vs Z 81/E 77) -> use Z
    "LOEI": ["Z", "E"],   # LOEI Z & E dead, only North works -> use N for all (z=e=n)
    "SKLT": ["E"],        # SKLT East dead -> use Z
    "IPM":  ["N"],        # MY.IPM North bad (verified 2026-06-06) -> Z-only uses Z anyway
}
BAD_STD_MIN = 5.0
# DET-only PZ override: NPW's per-station model was trained with a 2-zero (velocity) PZ, so feed
# its DET st_FV a 2-zero PZ so the features match training (magnitude still uses the 3-zero PZFILE).
DET_PZ = {"NPW": PROJECT + "/responses/pz_det/GE_NPW_HH.pz"}

def _fix_bad_components(out, sta):
    """Replace dead/bad components (manual map or flatline) with a good one (prefer Z, then N, then E)."""
    by = {tr.stats.channel[-1]: tr for tr in out}
    manual = set(BAD_COMPONENTS.get(sta, []))
    bad = [c for c in "ZNE" if c in manual or c not in by or float(np.std(by[c].data)) < BAD_STD_MIN]
    good = [c for c in "ZNE" if c not in bad and c in by]
    if not good or not bad:
        return out                        # all good, or unfixable (all bad) -> leave as is
    ref = by[good[0]].data                # ZNE order makes Z the preferred reference
    for c in bad:
        if c in by:
            by[c].data = ref.copy()
    return out

def get_window(sta, t1, t2):
    with LOCK[sta]:
        if sta not in BUFFERS or len(BUFFERS[sta]) == 0:
            return None
        st = BUFFERS[sta].slice(t1, t2).copy()
    out = _three_comp(st)
    return _fix_bad_components(out, sta) if out is not None else None

def get_z_window(sta, t1, t2):
    """Single-component (Z-only) window for the no-ML STA/LTA trigger + pick. Prefers Z; if Z is
    dead/flat or listed in BAD_COMPONENTS, falls back to the best alive of N/E. Returns a 1-trace
    Stream (or None). Lets us use any station with a healthy vertical, ignoring bad horizontals."""
    with LOCK[sta]:
        if sta not in BUFFERS or len(BUFFERS[sta]) == 0:
            return None
        st = BUFFERS[sta].slice(t1, t2).copy()
    by = {}
    for comp in ("Z", "N", "E"):
        sel = st.select(component=comp)
        if sel:
            for t_ in sel:                  # Step C-2 (2026-06-11): demean each contiguous segment BEFORE the zero-fill —
                if t_.data.size:            # raw counts carry DC offsets of 1e4-2e6, so a sub-10 s gap zero-filled at DC
                    t_.data = t_.data - np.median(t_.data)   # made false triggers (cft~10) + ~20 s post-gap LTA blindness
            tr = sel.merge(method=1, fill_value=0)[0]
            if hasattr(tr.data, "mask"):
                tr.data = tr.data.filled(0.0)
            by[comp] = tr
    if not by:
        return None
    manual = set(BAD_COMPONENTS.get(sta, []))
    for comp in ("Z", "N", "E"):                       # prefer Z; fall back to best alive component
        if comp in by and comp not in manual and float(np.std(by[comp].data)) >= BAD_STD_MIN:
            tr = by[comp]
            if abs(tr.stats.sampling_rate - SAMPLE_RATE) > 0.01:
                tr.resample(SAMPLE_RATE)
            tr.data = np.ascontiguousarray(tr.data, dtype=np.float64)
            return Stream([tr])
    return None

class SLClient(EasySeedLinkClient):
    def __init__(self, server, label):
        super().__init__(server, autoconnect=True)
        self.label = label
    def on_data(self, trace):
        ingest(trace)
    def on_seedlink_error(self):
        log(f"  SeedLink error ({self.label})")
    def on_terminate(self):
        log(f"  SeedLink terminate ({self.label})")

def server_thread(server, selects, label):
    while True:
        try:
            log(f"  connecting SeedLink {label} ({server}) — {len(selects)} channels")
            c = SLClient(server, label)
            for net, sta, sel in selects:
                c.select_stream(net, sta, sel)
            c.run()
        except Exception as e:
            log(f"  {label} client crashed: {e}; reconnect in 10 s")
            time.sleep(10)

def build_selects():
    out = {}
    for src in CFG.SOURCES.values():
        sel = []
        band = (src.get("channels", ["HH?"])[0])[:2]   # per-source band: "HH" (most) or "BH" (e.g. MY 20 sps)
        for sta in src["stations"]:
            if sta in DISABLED_STATIONS:
                continue
            for comp in ("Z", "N", "E"):
                sel.append((src["network"], sta, f'{_STA_LOC_OVERRIDE.get(sta, src["location"])}{band}{comp}'))
        if src["seedlink"] in out:            # MERGE sources sharing a SeedLink server (TM + MM on EarthScope)
            out[src["seedlink"]] = (out[src["seedlink"]][0] + sel, out[src["seedlink"]][1] + "+" + src["network"])
        else:
            out[src["seedlink"]] = (sel, src["network"])
    return out

# ----------------------------------------------------------------------------- event hub (associator)
HUB_LOCK = threading.Lock()
REPORTS  = []                 # list of dicts: station, P_utc, pick_wall, mags{tp:val}, snr
ACTIVE   = {"open": False, "t0_wall": None, "first_P": None, "located": False, "partial": False}

def submit_report(rep):
    with HUB_LOCK:
        REPORTS.append(rep)
        if not ACTIVE["open"]:
            ACTIVE.update(open=True, t0_wall=time.time(), first_P=rep["P_utc"], located=False, partial=False,
                          rescue_tried=False, rescue=None)              # Step D-2 state is per-cluster
        log(f"  HUB: report {rep['station']} P={rep['P_utc']} SNR10={rep.get('snr10',0):.1f} "
            f"SNR={rep['snr']:.1f}  (cluster {len(_cluster())})")
        _far_ingest(rep, time.time())     # Step H: feed the far-track associator (membership only — it can
                                          # never alert from here; alerts happen in its own 1-Hz tick AFTER
                                          # the primary has had precedence)

def _cluster():
    if ACTIVE["first_P"] is None:
        return []
    rows = [r for r in REPORTS if abs(r["P_utc"] - ACTIVE["first_P"]) <= ASSOC_WINDOW]
    best = {}
    for r in sorted(rows, key=lambda r: r["P_utc"]):   # Step C v2 (2026-06-11): ONE report per station (earliest
        best.setdefault(r["station"], r)               # P = the P; later ones are re-detections of the same onset
    return list(best.values())                         # or S) — duplicates must NEVER count toward 4-station
                                                       # coincidence / EDT pairs / amplitude-MAD (review blocker)

def _station_mag(r):
    return r["mags"].get(MAG_TP_DEFAULT) or (max(r["mags"].values()) if r["mags"] else 0.0)

# ---- FALSE-EVENT REJECTION GATES (research-backed, design study 2026-06-06) ----
# A noise burst made 4 clustered noisy NW stations declare a false M4.2 (one-sided -> RMS can't reject).
# Two gates, BOTH designed to still PASS real one-sided Myanmar/Laos events:
#  (1) SILENT-STATION (ShakeAlert EPIC "40% rule"): of the LIVE stations NEAR the candidate whose P
#      should already have arrived, >= SILENT_MIN_RATIO must have triggered. A real quake lights up its
#      neighbours; a local noise cluster leaves the rest of the near network silent. Magnitude-scaled
#      near-radius + a sparse-network floor so genuine sparse one-sided events are NOT penalised.
#  (2) AMPLITUDE-CONSISTENCY (AGREE): the picked stations' single-station Mwp must agree (low MAD).
SILENT_MIN_RATIO    = CFG.TRIGGER.get("silent_min_ratio", 0.70)   # TUNED on 230-event catalog: 0.70 rejects the 2026-06-06 false alarm (ratio 0.67) yet keeps 28/29 real located events (97%); next real event sits at 0.75
SILENT_GRACE        = 8.0       # s: predicted P must be this far in the past to count as "expected" (covers SeedLink latency)
SILENT_MIN_EXPECTED = 4         # sparse-network floor: <= this many expected -> skip gate (pass)
SILENT_STREAM_FRESH = 120.0     # s: a station counts as live if its last data is younger than this
SILENT_NEAR_MIN_KM  = 250.0     # near-radius floor
SILENT_NEAR_MAX_KM  = 550.0     # near-radius cap (don't expect far stations to detect)
AMP_MAD_MAX         = CFG.TRIGGER.get("amp_mad_max", 1.0)   # lenient backstop (real Mwp MAD ~0.2-0.5); silent-station gate is the primary discriminator
# MwP-station SNR10 gate (2026-06-06): the magnitude (closest calibrated) station must show a
# clear signal. SNR10 = RMS(P..P+10)/RMS(P-10..P) on Z, 2-8 Hz. TUNED on the catalog: real located events
# median 53.8 (only a 1.03 anomaly below); the false alarm's MwP station (CMMT) = 2.37; next real = 3.08.
# 2.5 rejects the false alarm + keeps 28/29 real (97%); SNR>=5 would kill a real M5.1. Second backstop only.
MWP_SNR_MIN         = CFG.TRIGGER.get("mwp_snr_min", 2.5)
# MAG-FRESHNESS gate (TEST 2 / KAPI phantom, 2026-06-21): the named magnitude station must have FRESH,
# on-scale data covering its Mwp window AT ALERT TIME. The 06-14 phantom took its magnitude from KAPI while
# KAPI was OFFLINE (its SeedLink feed had stalled): the buffer still held an OLD real arrival, so the Mwp/SNR
# path produced a high, gate-passing magnitude from data that was actually days stale. Nothing in the chain
# verified that LAST_DATA[mag_sta] reached past the Mwp window. We require: (a) the mag station's last data is
# fresher than MAG_FRESH_SEC relative to NOW, AND (b) its last data actually covers P+10 (the Mwp window).
# Real events ALWAYS satisfy this (we just picked & measured the window from live data seconds ago); only a
# stale/offline buffer fails -> ~zero real-recall loss, blocks the phantom at the source.
MAG_FRESH_SEC       = CFG.TRIGGER.get("mag_fresh_sec", 600.0)   # last data must be younger than this at alert time
NEAR_FIELD_KM       = CFG.TRIGGER.get("near_field_km", 800.0)   # B/near-field-support (2026-06-14): for a SUB-great
# event the NEAREST locating station must be within this of the epicentre. A real in-grid quake has near support;
# a teleseism mislocated to a station-free cell (Luzon Strait/SCS/Vietnam in the 8-day replay, nearest >=1067 km)
# does NOT. Mirrors coherent_subset SUBSET_NEAR_KM=600 on the PRIMARY path (which lacked it). Great events
# (>=FAR_MAG) EXEMPT (alert anywhere). Real worst case = M7.8 Mindanao nearest 594 km < 800 -> passes.
# E/triangulation-hole gate (2026-06-21, harden #2): suppress a SUB-great (M<FAR_MAG) primary-path alert when the
# epicentre falls in the EASTERN/SOUTHERN station-free "hole" (HOLE_BOX) AND the NEAREST locating station is farther
# than HOLE_NEAR_KM. Inside this box a teleseism can mislocate into a grid cell with no nearby support and the existing
# 800 km near-field gate (B) is too loose to catch it (kills +15 box phantoms the 800 km gate misses). Catalog red-team
# (USGS+EMSC 1990-2026, 4318 in-box events): real M[4.5,6.5) events with nearest>400 km = 9 USGS (+~2 EMSC SCS) in 36.5 yr
# = 0.25-0.30/yr, ALL >=412 km from the network and 660-1364 km from Thailand (Vietnam La Gi swarm + Borneo/SCS interior)
# -> outside the regional EEW envelope. The DENSE Java/Sumatra subduction band (2358 events) is 100%% within 400 km ->
# NEVER touched. Great events (>=FAR_MAG, e.g. the 27 in-box M>=6.5 incl. 2007 M8.4) are EXEMPT (alert anywhere).
HOLE_BOX     = (-7.0, 16.0, 101.0, 116.0)   # (minlat, maxlat, minlon, maxlon)
HOLE_NEAR_KM = CFG.TRIGGER.get("hole_near_km", 400.0)
def _in_hole_box(loc):
    a, b, c, d = HOLE_BOX
    return a <= loc["lat"] <= b and c <= loc["lon"] <= d
ALERT_MAX_LATENCY   = CFG.TRIGGER.get("alert_max_latency_s", 360.0)   # D/slow-alert (2026-06-14): a SUB-great event
# must alert within this of its ORIGIN. A real local quake alerts in <=~5 min (USGS in-grid sim n=401: sub-great
# MAX 4.8 min); a SCATTERED-noise cluster takes far longer (the 2 live false alarms were 7.1 & 7.7 min — picks NOT
# on the moveout). Great events (>=FAR_MAG) EXEMPT (the 600 s stale gate still covers them). Complements B: B
# rejects FAR-located mislocations, D rejects SLOW near-located scattered-pick false alarms.
# Per-pick SNR gate (2026-06-06): a P counts toward the 3/4-station coincidence ONLY if its
# 10 s SNR (RMS(P..P+10)/RMS(P-10..P), Z, 2-8 Hz) >= this. Drops noise picks at the source (stops the
# 3-station-incomplete spam + prevents noise from building a false event). Tunable; validate recall.
PICK_SNR_MIN        = CFG.TRIGGER.get("pick_snr_min", 2.0)
# Per-station STA/LTA ON threshold (tune trigger for noisy stations only). Empirical
# (noise_why.py): MHIT/GSI/CMAI/PRAC over-trigger; raise their ON so chronic transients don't fire.
STALTA_ON_BY        = CFG.TRIGGER.get("stalta_on_by", {"MHIT": 5.5, "GSI": 5.5, "CMAI": 5.5, "PBKT": 5.0, "PRAC": 5.0, "SKLT": 5.0})

def _near_radius_km(mag):
    """Magnitude-scaled 'nearby' radius (EPIC-style), clipped: small events only expected close in."""
    r = 150.0 * (10.0 ** (0.5 * (max(mag, 3.5) - 3.5)))     # M3.5->150, M4.5->474, >=M5.5 capped
    return float(np.clip(r, SILENT_NEAR_MIN_KM, SILENT_NEAR_MAX_KM))

def _mwp_snr10(sta, p_arrival):
    """SNR10 of the magnitude station: RMS(P..P+10)/RMS(P-10..P) on Z, 2-8 Hz (the user's definition).
    Independent waveform backstop to the geometry-based silent-station gate. 0.0 if not computable."""
    p_arrival = UTCDateTime(p_arrival)   # hub passes report["P_utc"] as a float epoch; obspy 1.4 slice() needs UTCDateTime
    z = get_z_window(sta, p_arrival - 12, p_arrival + 11)
    if not z:
        return 0.0
    try:
        t = z[0].copy(); t.detrend("demean")
        t.filter("bandpass", freqmin=DET_BAND[0], freqmax=DET_BAND[1], corners=4, zerophase=True)
        pk = UTCDateTime(p_arrival)
        sig = t.slice(pk, pk + 10).data; noi = t.slice(pk - 10, pk).data
        if len(sig) > 20 and len(noi) > 20 and np.std(noi) > 0:
            return float(np.std(sig) / np.std(noi))
    except Exception:
        pass
    return 0.0

def mag_station_fresh(sta, p_arrival, live=None, now=None):
    """TEST 2 / KAPI phantom gate. Return (fresh, age_s, covers) for the NAMED magnitude station:
      fresh  = its feed is live (last data younger than MAG_FRESH_SEC at alert time `now`),
      covers = its last data actually reaches past the Mwp window (P + 10 s) so the magnitude was
               measured from data that EXISTS at alert time (not a stale buffer holding an old arrival).
    A real event always passes (we picked + measured that window from data that streamed in seconds ago);
    only a stale/offline buffer (the 06-14 KAPI hole) fails. `live`/`now` injectable for replay tests."""
    now = UTCDateTime(now) if now is not None else UTCDateTime()
    p_end = UTCDateTime(p_arrival) + 10.0                       # the Mwp window end (P..P+10)
    if live is not None:                                       # replay: live is a dict {sta: last_data UTC} or a set
        ld = live.get(sta) if isinstance(live, dict) else (now if sta in live else None)
    else:
        ld = LAST_DATA.get(sta)
    if ld is None:
        return False, 1e9, False
    age = float(now - ld)
    fresh  = age <= MAG_FRESH_SEC
    covers = ld >= p_end                                       # data must extend through the measured window
    return (fresh and covers), age, covers

def silent_station_check(loc, origin_utc, triggered_stas, mag, live=None, now=None):
    """EPIC 40% rule. live=None -> use LAST_DATA (live pipeline); pass live set + now for replay tests.
    Returns (ok, ratio, n_triggered, n_expected, n_lagging_feeds) — nlag>0 lets the hub DEFER the verdict (v3)."""
    _ensure_tt()
    now = now if now is not None else UTCDateTime()
    nearR = _near_radius_km(mag)
    nlag = 0
    triggered = set(triggered_stas)
    expected = set(triggered)                       # triggered stations are expected by definition
    for s, c in STATION_COORDS.items():
        if s in expected:
            continue
        if live is not None:
            if s not in live:
                continue
        else:
            ld = LAST_DATA.get(s)
            if ld is None or (now - ld) > SILENT_STREAM_FRESH:
                continue                            # not currently streaming -> can't be "expected"
        Rkm = gps2dist_azimuth(loc["lat"], loc["lon"], c["lat"], c["lon"])[0] / 1000.0
        if Rkm > nearR:
            continue                                # too far to be expected for this magnitude
        pP = origin_utc + float(np.interp(_dist_deg(loc["lat"], loc["lon"], np.array([c["lat"]]), np.array([c["lon"]]))[0], TT_DIST, TT_TIME))
        if (now - pP) < SILENT_GRACE:
            continue                                # its P hasn't had time to arrive+pick yet
        if live is None:
            ldc = LAST_DATA.get(s)
            if ldc is None or ldc < pP + 13.0:      # v3: count lagging feeds (data not yet past P+pick window) — but
                nlag += 1                           # REPORT them so the hub can DEFER the verdict until their data
        expected.add(s)                             # arrives. Exclusion weakened the gate (re-review B-1); deferral
                                                    # keeps the validated 2026-06-06 rejection AND fixes the real-event miss.
    n_exp = len(expected)
    if n_exp <= SILENT_MIN_EXPECTED:
        return True, 1.0, len(triggered), n_exp, 0  # sparse -> pass (preserve genuine one-sided)
    ratio = len(triggered & expected) / float(n_exp)
    return (ratio >= SILENT_MIN_RATIO), ratio, len(triggered), n_exp, nlag

def amplitude_consistency_check(reps, loc):
    """AGREE-style: single-station Mwp across the picked stations must agree (MAD <= AMP_MAD_MAX)."""
    Ms = []
    for r in reps:
        if r.get("cl", {}).get("clipped"):              # railed channel under-reads Mwp -> exclude so it can't veto a real event
            continue
        c = STATION_COORDS.get(r["station"])
        if not c:
            continue
        Rkm = gps2dist_azimuth(loc["lat"], loc["lon"], c["lat"], c["lon"])[0] / 1000.0
        m = _classical_M(r.get("cl", {}), Rkm, r["station"]).get("Mwp", 0.0)
        if m and m > 0:
            Ms.append(m)
    if len(Ms) < 3:
        return True, 0.0                            # too few to judge -> don't veto
    Ms = np.array(Ms)
    return (float(np.median(np.abs(Ms - np.median(Ms)))) <= AMP_MAD_MAX), float(np.median(np.abs(Ms - np.median(Ms))))

# ---- Thailand-relevance gate (2026-06-06) -----------------------------------------------
# With the regional expansion (Andaman -> eastern Indonesia/Papua), the network can locate quakes in
# very-active far zones (Banda/Sulawesi/Molucca) that are irrelevant to Thailand. Keep those stations for
# detection/confirmation of GREAT events but DON'T flood the public channel: alert publicly ONLY if the
# epicenter is within RELEVANCE_KM of Thailand, OR magnitude >= FAR_MAG (great events anywhere still go out).
THAI_REFS = [(18.80, 98.98), (15.28, 105.47), (13.70, 100.50), (7.88, 98.39), (6.42, 101.82)]  # N, NE, central, Andaman(Phuket), far-S
RELEVANCE_KM = 1800.0   # public-alert if epicenter within this of Thailand (announced on the channel)
FAR_MAG      = 6.5      # ...OR magnitude >= this anywhere (great events, e.g. 2004 M9.1)
SP_REFINE_ON = os.environ.get("SP_REFINE", "1") == "1"   # ≥4P+≥1 consensus-S re-locate before the single alert (sp_refine.py); SP_REFINE=0 -> P-only
# REQUIRE ≥1 consensus S before the single text alert (2026-06-16): the alert WAITS (bounded, adaptive
# to the nearest station's predicted S) so the posted epicentre + origin time are S-refined; a P-only SAFETY fallback
# fires at the cap so a confirmed event is NEVER lost; great/tsunami events (bigflag) fire IMMEDIATELY (speed > refinement).
SP_REQUIRE_S     = os.environ.get("SP_REQUIRE_S", "1") == "1"   # SP_REQUIRE_S=0 -> old behaviour (fire on ≥4P, S-refine when available)
SP_S_WAIT_MARGIN = 15.0    # s past the predicted nearest-station S to keep retrying (SeedLink latency + PhaseNet + jitter)
SP_S_WAIT_MIN    = 18.0    # s floor on the per-event S-wait cap (close events deliver an S fast)
SP_S_WAIT_MAX    = 90.0    # s ceiling (far events fire P-only here; << ALERT_MAX_LATENCY 360 s and the 180 s auto-close)
FASTPATH_AZ_GAP  = 180.0   # GOOD-GEOMETRY P-ONLY FAST PATH (2026-06-20, deployed): skip the S-wait when the locate is
FASTPATH_RMS     = 1.5     # well-surrounded (az_gap<=this) AND tightly coherent (rms<=this). Validated n=238/1283 events
                           # -> P-only location median 14 km / 90% 69 km, ZERO accuracy cost vs S-refine; gates all upstream.
def _nearest_locating_km(loc, reps):
    """Distance (km) from the epicentre to the NEAREST locating station — near-field-support discriminant (B).
    A real local quake always has a station reasonably close; a far teleseism mislocated into a station-free
    grid cell does not (all its support is >1000 km away)."""
    ds = [gps2dist_azimuth(loc["lat"], loc["lon"], STATION_COORDS[r["station"]]["lat"], STATION_COORDS[r["station"]]["lon"])[0] / 1000.0
          for r in reps if r["station"] in STATION_COORDS]
    return min(ds) if ds else 1e9

def _min_thai_km(loc):
    return min(gps2dist_azimuth(loc["lat"], loc["lon"], la, lo)[0] / 1000.0 for la, lo in THAI_REFS)
def thailand_relevant(loc, mag):
    return (_min_thai_km(loc) <= RELEVANCE_KM) or (mag >= FAR_MAG)

# ---- spatial-coherence SUBSET associator (RANSAC seed-and-grow) — rescues a real event drowned in CONCURRENT
# noise picks. The M4.0 Myanmar (TMD 16715, 2026-06-13) was MISSED: the earliest-4 picks were Indonesian noise
# -> grid-edge (RMS 138s) -> reject; the drop-1 D-2 rescue can't survive MULTIPLE noise picks. This finds the
# LARGEST hypocentre-consistent subset (right pick per station), so {NGU,TGI,MHIT,CMAI,CMMT...} is selected and
# the noise dropped. Validated on the REAL 16715 stream: 21.66,96.06 RMS 0.61s, origin TMD+2s, 33 km.
# PRODUCTION-SAFE (review 2026-06-14): the scan uses a FAST coarse-grid locate and runs in a BACKGROUND thread
# (lock-free), so it never stalls the 1-Hz hub; the final subset gets the full production locate.
SUBSET_WALL_WIN  = 240.0    # s: search picks reported within this wall window
SUBSET_MIN       = 5        # require >=5 coherent inliers (false-assembly guard; the full gate chain still applies)
SUBSET_THROTTLE  = 6.0      # s: re-trigger the background search at most this often per open cluster
SUBSET_HISNR     = 8.0      # only run when a real-event signature is present: >=SUBSET_HISNR_N recent picks this strong
SUBSET_HISNR_N   = 2        # (a buried REAL quake lights up strong near picks; pure noise does not -> no idle spinning)
SUBSET_MAX_RMS   = 1.5      # the rescue is a FALLBACK -> accept only a CONFIDENT subset (RMS<=this, vs 4.0 for first-4):
                            # real 16715 rescue had RMS 0.61; marginal one-sided edge sets (1.7-3.5 RMS) are rejected
SUBSET_MAX_KM    = 1200.0   # ...and only in the NEAR FIELD, where the Thai network geometry makes the rescue reliable
                            # (16715 Myanmar = 440 km). Far/edge events buried in noise locate badly (one-sided, ~400 km
                            # error) -> not rescued here (the far-track handles great events; USGS handles the rest).
SUBSET_NEAR_KM   = 600.0    # ...and a contributing station must be within this of the epicentre. A REAL near event is
                            # DETECTED because a station is right there (16715 nearest = 128 km); a phantom located
                            # OUTSIDE the network by distant one-sided picks has none (the 1-per-4.4-day backtest false
                            # alarm: epicentre 1159 km from its nearest station, and its noise Mwp's were coincidentally
                            # consistent so the amplitude-MAD gate did NOT catch it — review/backtest 2026-06-14).
_SUBSET_RUNNING  = [False]  # a background search is in flight
_SUBSET_RESULT   = [None]   # (loc, inliers, t0) awaiting application by the hub, or None
_SUBSET_GEN      = [0]      # cluster generation token: a worker only publishes if the cluster that spawned it is
                            # still open (incremented on every _reset_event) -> a stale result can't apply to a NEW
                            # cluster and fire a misattributed alert (review 2026-06-14)

def _fast_locate(arrivals, step=1.0):
    """Fast COARSE-grid-only EDT locate (no fine refinement) for the subset SCAN — ~1-3 ms vs ~30 ms full."""
    stas = [a for a in arrivals if a["station"] in STATION_COORDS]
    if len(stas) < MIN_STATIONS:
        return None
    obs = np.array([a["relative_time"] for a in stas])
    slat = np.array([STATION_COORDS[a["station"]]["lat"] for a in stas])
    slon = np.array([STATION_COORDS[a["station"]]["lon"] for a in stas])
    ia, ib = np.triu_indices(len(stas), k=1)
    dT = obs[ia] - obs[ib]; s2 = 2.0 * (EDT_SIGMA ** 2)
    las = np.arange(GRID_LAT[0], GRID_LAT[1] + 0.01, step)
    los = np.arange(GRID_LON[0], GRID_LON[1] + 0.01, step)
    LA, LO = np.meshgrid(las, los, indexing="ij")
    PRED = np.stack([np.interp(_dist_deg(LA, LO, float(slat[k]), float(slon[k])), TT_DIST, TT_TIME) for k in range(len(stas))])
    RES = dT[:, None, None] - (PRED[ia] - PRED[ib])
    L = np.exp(-(RES * RES) / s2).sum(axis=0)
    jla, jlo = np.unravel_index(int(np.argmax(L)), L.shape)
    la, lo = float(las[jla]), float(los[jlo])
    pred = np.interp(_dist_deg(la, lo, slat, slon), TT_DIST, TT_TIME)
    ot = float(np.median(obs - pred)); rms = float(np.sqrt(np.mean((obs - (ot + pred)) ** 2)))
    return {"lat": la, "lon": lo, "rms": rms, "origin_time": ot}

def coherent_subset(reports, max_rms=None, min_sta=None, resid_tol=8.0, topn=8, per_sta=2, final_rms=None):
    """Largest hypocentre-consistent subset -> (loc, inlier_reports, t0) or (None,None,None). Lock-free + thread-
    safe (reads only read-only tables). SCAN uses _fast_locate (loose max_rms filter); the FINAL inlier set gets
    the full production locate_earthquake and must satisfy the TIGHT final_rms (a confident fallback only)."""
    import itertools
    if max_rms is None: max_rms = MAX_RMS_LOCATE
    if min_sta is None: min_sta = MIN_STATIONS
    if final_rms is None: final_rms = SUBSET_MAX_RMS
    by_sta = {}
    for r in reports:
        if r["station"] in STATION_COORDS:
            by_sta.setdefault(r["station"], []).append(r)
    if len(by_sta) < min_sta:
        return None, None, None
    cands = []
    for s, lst in by_sta.items():
        lst.sort(key=lambda r: -r.get("snr10", 0.0)); cands.extend(lst[:per_sta])   # strong real pick != earliest
    cands.sort(key=lambda r: -r.get("snr10", 0.0)); seeds = cands[:topn]            # C(8,4)=70 seed combos
    def predict(loc, origin, s):
        c = STATION_COORDS.get(s)
        if not c: return None
        d = float(_dist_deg(loc["lat"], loc["lon"], np.array([c["lat"]]), np.array([c["lon"]]))[0])
        return origin + float(np.interp(d, TT_DIST, TT_TIME))
    best = None
    for combo in itertools.combinations(seeds, min_sta):
        if len({c["station"] for c in combo}) < min_sta: continue
        t0 = min(c["P_utc"] for c in combo)
        loc = _fast_locate([{"station": c["station"], "relative_time": c["P_utc"] - t0} for c in combo])  # cheap filter
        if not loc or loc["rms"] > max_rms: continue
        origin = t0 + loc["origin_time"]
        inl = {}
        for s, lst in by_sta.items():                     # GROW: each station's best pick fitting the prediction
            pred = predict(loc, origin, s)
            if pred is None: continue
            fit = [r for r in lst if abs(r["P_utc"] - pred) <= resid_tol]
            if fit: inl[s] = min(fit, key=lambda r: abs(r["P_utc"] - pred))
        if len(inl) < min_sta: continue
        t0b = min(r["P_utc"] for r in inl.values())
        loc2 = locate_earthquake([{"station": s, "relative_time": r["P_utc"] - t0b} for s, r in inl.items()])  # full, precise
        if not loc2 or loc2["rms"] > final_rms: continue            # TIGHT acceptance — confident rescue only
        if _min_thai_km(loc2) > SUBSET_MAX_KM: continue             # near-field only (reliable geometry)
        _near = min(float(_dist_deg(loc2["lat"], loc2["lon"], np.array([STATION_COORDS[s]["lat"]]),
                                    np.array([STATION_COORDS[s]["lon"]]))[0]) * 111.0 for s in inl)
        if _near > SUBSET_NEAR_KM: continue                         # epicentre OUTSIDE the network = phantom -> reject
        score = (len(inl), -loc2["rms"])
        if best is None or score > best[0]:
            best = (score, loc2, list(inl.values()), t0b)
    if best:
        return best[1], best[2], best[3]
    return None, None, None

def _subset_worker(snapshot, gen):
    """Run the (lock-free) subset search OFF the hub thread, then publish the result under the lock — but ONLY if
    the cluster that spawned it is still the current one (gen token), else drop it (review 2026-06-14: a stale
    result from a closed cluster must never apply to the next cluster and fire a misattributed alert)."""
    res = (None, None, None)
    try:
        res = coherent_subset(snapshot)
    except Exception:
        log("SUBSET worker error:\n" + traceback.format_exc())
    finally:
        with HUB_LOCK:
            if gen == _SUBSET_GEN[0]:
                _SUBSET_RESULT[0] = res if res[0] else None
            _SUBSET_RUNNING[0] = False

def hub_loop():
    while True:
        time.sleep(1.0)
        with HUB_LOCK:
            if not ACTIVE["open"]:
                continue
            cl = _cluster()
            good = [r for r in cl if r.get("snr", 0) >= MIN_SNR_LOCATE]   # only real-signal picks may locate
            # FAST: >=4 stations with SNR>=MIN_SNR_LOCATE -> locate (noise triggers are SNR<2 -> excluded)
            # enter on a narrow >=4 cluster (fast path) OR when the broad recent window is rich enough for a
            # subset rescue (the 16715 case: the earliest pick anchored a 3-station noise cluster, but the real
            # event is recoverable from the wider window).
            _broad_n = len({r["station"] for r in REPORTS if time.time() - r.get("pick_wall", time.time()) <= SUBSET_WALL_WIN})
            if not ACTIVE["located"] and (len(good) >= MIN_STATIONS or _broad_n >= SUBSET_MIN):
                ordered = sorted(good, key=lambda r: r["P_utc"])
                loc = None; first4 = ordered[:4]; t0 = ordered[0]["P_utc"] if ordered else time.time()
                if len(good) >= MIN_STATIONS:                        # narrow cluster present -> the fast first-4 EDT path
                    t0 = min(r["P_utc"] for r in first4)
                    arrivals = [{"station": r["station"], "relative_time": float(r["P_utc"] - t0)} for r in first4]
                    loc = locate_earthquake(arrivals)
                if not (loc and loc["rms"] <= MAX_RMS_LOCATE):
                    # Step D-2 (2026-06-11): the first-4 locate FAILED (RMS>4 or grid-edge) — the event would be
                    # lost. ONE leave-one-out rescue round over the first 5 picks; an accepted candidate stays
                    # PENDING until an independent pick confirms it (false-assembly ~0.1-0.9/yr at current rates).
                    if loc:
                        log(f"  hub: RMS {loc['rms']:.1f}s > {MAX_RMS_LOCATE:.0f}s — picks not hypocentre-consistent, rejected (likely noise)")
                    if not ACTIVE.get("rescue_tried") and len(ordered) >= MIN_STATIONS + 1:
                        ACTIVE["rescue_tried"] = True
                        _resc = _loo_rescue(ordered)
                        if _resc:
                            ACTIVE["rescue"] = _resc
                            log("  D-2 RESCUE CANDIDATE: dropped %s (residual %.1fs), subset RMS %.2fs at %.2f,%.2f — PENDING independent confirmation"
                                % (_resc["excluded"], _resc["exc_resid"], _resc["loc"]["rms"], _resc["loc"]["lat"], _resc["loc"]["lon"]))
                    loc = None
                    if ACTIVE.get("rescue"):
                        _conf = _rescue_confirmed(ACTIVE["rescue"], cl)
                        if _conf:
                            loc = ACTIVE["rescue"]["loc"]
                            first4 = ACTIVE["rescue"]["subset"]
                            t0 = ACTIVE["rescue"]["t0"]
                            ordered = [r for r in ordered if r["station"] != ACTIVE["rescue"]["excluded"]]
                            good = [r for r in good if r["station"] != ACTIVE["rescue"]["excluded"]]   # and from the silent/MAD gates
                            log("  D-2 RESCUE CONFIRMED by %s — proceeding to the standard gate chain (excluded %s also barred from magnitude)"
                                % (_conf, ACTIVE["rescue"]["excluded"]))
                    # SUBSET RESCUE (2026-06-14, production-safe): earliest-4 + drop-1 D-2 both failed -> a real
                    # quake may be buried in concurrent noise (the 16715 miss). Trigger a BACKGROUND, lock-free
                    # search (only when a real-event signature is present: >=SUBSET_HISNR_N strong recent picks),
                    # and APPLY a completed search result on a later tick. The search NEVER runs on the hub thread.
                    if (loc is None and _SUBSET_RESULT[0] is None and not _SUBSET_RUNNING[0]
                            and (time.time() - ACTIVE.get("subset_last", 0.0)) >= SUBSET_THROTTLE):
                        _rec = [r for r in REPORTS if time.time() - r.get("pick_wall", time.time()) <= SUBSET_WALL_WIN]
                        if (sum(1 for r in _rec if r.get("snr10", 0) >= SUBSET_HISNR) >= SUBSET_HISNR_N
                                and len({r["station"] for r in _rec}) >= SUBSET_MIN):
                            ACTIVE["subset_last"] = time.time(); _SUBSET_RUNNING[0] = True
                            threading.Thread(target=_subset_worker, args=([dict(r) for r in _rec], _SUBSET_GEN[0]), daemon=True).start()
                    if loc is None and _SUBSET_RESULT[0] is not None:
                        _sl, _sin, _st0 = _SUBSET_RESULT[0]; _SUBSET_RESULT[0] = None
                        if _sl and len(_sin) >= SUBSET_MIN:
                            loc = _sl; t0 = _st0; first4 = sorted(_sin, key=lambda r: r["P_utc"])
                            ordered = list(first4); good = list(first4)   # gates + magnitude run on the coherent set
                            log("  SUBSET RESCUE: largest coherent set = %d stations %s, RMS %.2fs at %.2f,%.2f — earliest-4 was noise-contaminated"
                                % (len(_sin), sorted(r["station"] for r in _sin), _sl["rms"], _sl["lat"], _sl["lon"]))
                if loc and loc["rms"] <= MAX_RMS_LOCATE:
                    # --- S-P REFINEMENT (≥4P + ≥1 consensus-S single-alert design, 2026-06-15) ------------------
                    # Re-locate epicentre + origin with a 2-of-2 PhaseNet stead+diting consensus S (cons+residdrop)
                    # BEFORE the single alert (no repeat post). Off-critical-path picker on the live BUFFERS;
                    # returns None -> P-only fallback (no S in the buffers yet). Validated offline: median loc err
                    # 83->42 km, hard cases 165->82 km, 76 better / 6 worse. SP_REFINE=0 disables.
                    _sp_origin = t0 + loc["origin_time"]; _has_S = False
                    if SP_REFINE_ON:
                        try:
                            import sp_refine
                            _pp = {r["station"]: float(r["P_utc"]) for r in first4}
                            _ref = sp_refine.refine(_pp, {"lat": loc["lat"], "lon": loc["lon"]}, BUFFERS,
                                                    STATION_COORDS, TT_DIST, TT_TIME, TT_S, log)
                            if _ref:
                                loc["lat"], loc["lon"] = _ref["lat"], _ref["lon"]
                                loc["sp_n"] = _ref["n_s"]; _sp_origin = _ref["origin"]; _has_S = True
                                loc["sp_s"] = _ref.get("s_times", {})    # {sta: S abs time} -> green S markers on the plots
                                loc["sp_p"] = _ref.get("p_times", {})    # {sta: refined PhaseNet P} -> red P markers (Decision B)
                        except Exception:
                            log("  sp_refine error:\n" + traceback.format_exc())
                    origin = UTCDateTime(_sp_origin)                 # joint P+S origin if refined, else P-only EDT
                    # MAGNITUDE = network MEDIAN over the CALIBRATED good-pick stations in the located coherent set
                    # (2026-06-15, replaces single-fastest -> glitch-robust). Mis-picked stations are already excluded
                    # from `ordered` by the coherence/RMS locate, so the magnitude rides the SAME good-pick set the
                    # location trusts. NPW-type locate-only stations are dropped (no own Mwp term). mbP small-event
                    # cross-check is applied inside combine_magnitude. Single fallback if nothing is clean. Cap 8 =
                    # generous bound (the coherent set is small); uses ALL calibrated good picks available, no 1/3/4 cap.
                    _cal = [r for r in ordered if r["station"] in MAG_CALIBRATED]
                    _cal_clean = [r for r in _cal if r.get("cl", {}).get("mint") and not r.get("cl", {}).get("clipped")]   # on-scale, complete, NOT gapped (gap -> no mint)
                    _cal_cl = [r for r in _cal if r.get("cl")]                  # v3: a locate-only report (cl={}, Mwp window
                    _any_cl = [r for r in ordered if r.get("cl")]               # uncovered) must NEVER be the magnitude station
                    mag_reps = _cal_clean[:8] or _cal_cl[:8] or _any_cl[:1] or ordered[:1]
                    Mfinal, mlabel, bigflag, mdetail = combine_magnitude(mag_reps, loc)
                    ok_sil, sratio, ntr, nexp, nlagf = silent_station_check(loc, origin, [r["station"] for r in good], Mfinal)
                    if (not ok_sil) and nlagf > 0 and (time.time() - ACTIVE["t0_wall"]) < 45.0:
                        log("  silent-gate DEFER: %d lagging feed(s) have not delivered their P window yet — re-checking (%.0f s into event)"
                            % (nlagf, time.time() - ACTIVE["t0_wall"]))
                        continue                    # v3 (review B-1): wait for slow feeds instead of weakening the gate;
                                                    # the verdict re-runs each second until data arrives or 45 s elapse
                    ok_amp, amad = amplitude_consistency_check(good, loc)
                    snr_mwp = _mwp_snr10(mag_reps[0]["station"], mag_reps[0]["P_utc"]) if mag_reps else 0.0
                    mfresh, mage, mcov = mag_station_fresh(mag_reps[0]["station"], mag_reps[0]["P_utc"]) if mag_reps else (False, 1e9, False)
                    if not ok_sil:
                        log("  REJECT silent-station: only %d/%d nearby live stations triggered (%.0f%% < %.0f%%) -> correlated noise, not an event"
                            % (ntr, nexp, 100*sratio, 100*SILENT_MIN_RATIO))
                    elif not mfresh:
                        log("  REJECT mag-stale: magnitude station %s data %.0fs old / covers-window=%s (need <%.0fs + past P+10) "
                            "-> offline/stale buffer, magnitude not trustworthy (KAPI phantom guard) — alert withheld"
                            % (mag_reps[0]["station"], mage, mcov, MAG_FRESH_SEC))
                    elif snr_mwp < MWP_SNR_MIN:
                        log("  REJECT mwp-snr: magnitude station %s SNR10=%.2f < %.1f (P..P+10 vs pre-P, 2-8Hz) -> no clear signal, likely noise"
                            % (mag_reps[0]["station"], snr_mwp, MWP_SNR_MIN))
                    elif not ok_amp:
                        log("  REJECT amplitude-consistency: single-station Mwp MAD %.2f > %.2f -> scattered amplitudes, likely noise"
                            % (amad, AMP_MAD_MAX))
                    elif not thailand_relevant(loc, Mfinal):
                        log("  SUPPRESS far event: epicenter %.0f km from Thailand, M%.1f (< %.1f) -> real quake but not Thailand-relevant, public alert withheld"
                            % (_min_thai_km(loc), Mfinal, FAR_MAG))
                    elif Mfinal < FAR_MAG and _nearest_locating_km(loc, good) > NEAR_FIELD_KM:
                        log("  SUPPRESS no near-field support: nearest locating station %.0f km > %.0f km (M%.1f < %.1f) -> teleseism / remote mislocation, not a local event"
                            % (_nearest_locating_km(loc, good), NEAR_FIELD_KM, Mfinal, FAR_MAG))
                    elif Mfinal < FAR_MAG and _in_hole_box(loc) and _nearest_locating_km(loc, good) > HOLE_NEAR_KM:
                        log("  SUPPRESS triangulation-hole: epicentre in station-free box %s, nearest locating station %.0f km > %.0f km (M%.1f < %.1f) -> mislocated teleseism in a hole, not a local event"
                            % (str(HOLE_BOX), _nearest_locating_km(loc, good), HOLE_NEAR_KM, Mfinal, FAR_MAG))
                    elif Mfinal < FAR_MAG and float(UTCDateTime() - origin) > ALERT_MAX_LATENCY:
                        log("  SUPPRESS slow alert: origin %.0f s ago > %.0f s (M%.1f < %.1f) -> picks not on the moveout (scattered noise), not a fast local event"
                            % (float(UTCDateTime() - origin), ALERT_MAX_LATENCY, Mfinal, FAR_MAG))
                    elif float(UTCDateTime() - origin) > 600.0:
                        log("  SUPPRESS stale event: origin %.0f s old (SeedLink backlog replay after an outage?) — public alert withheld (v2, review F5)"
                            % float(UTCDateTime() - origin))
                    elif Mfinal <= 0:
                        log("  SUPPRESS zero-magnitude: no station had a complete Mwp window (all locate-only) — public alert withheld (v3, review A-1)")
                    else:
                        # REQUIRE ≥1 consensus S (2026-06-16): every false-alarm gate passed -> this is a
                        # CONFIRMED real event. Hold (bounded, adaptive to the nearest locating station's predicted S)
                        # so the posted epicentre + origin time are S-refined. Same proven re-entrant pattern as the
                        # silent-gate DEFER above. Fire P-only as a SAFETY fallback at the cap (a confirmed event is
                        # NEVER lost), and fire IMMEDIATELY for great/tsunami events (bigflag) where speed > refinement.
                        # FAST PATH (2026-06-20, deployed): a well-surrounded + tightly-coherent locate is already
                        # accurate (validated ~14 km) -> skip the S-wait, fire P-only NOW. Same early-fire path bigflag
                        # already uses; forgoes only S-refinement. ALL false-alarm gates ran above -> no new FA risk.
                        _good_geom = (loc.get("az_gap", 360.0) <= FASTPATH_AZ_GAP and loc.get("rms", 9.9) <= FASTPATH_RMS)
                        if _good_geom and not _has_S:
                            log("  FAST-PATH: good geometry (az_gap %.0f<=%.0f, RMS %.2f<=%.1f) -> fire P-only NOW, skip S-wait (validated ~14km)"
                                % (loc.get("az_gap", 360.0), FASTPATH_AZ_GAP, loc.get("rms", 9.9), FASTPATH_RMS))
                        if SP_REQUIRE_S and SP_REFINE_ON and not _has_S and not bigflag and not _good_geom:
                            _evage = float(UTCDateTime() - origin)
                            _cd = min((gps2dist_azimuth(loc["lat"], loc["lon"], _c["lat"], _c["lon"])[0] / 1000.0
                                       for _c in (STATION_COORDS.get(r["station"]) for r in first4) if _c), default=300.0)
                            _scap = min(max(float(np.interp(_cd / 111.19, TT_DIST, TT_S)) + SP_S_WAIT_MARGIN,
                                            SP_S_WAIT_MIN), SP_S_WAIT_MAX)
                            if _evage < _scap:
                                if not ACTIVE.get("s_wait_logged"):
                                    log("  S-WAIT: confirmed real event — holding for ≥1 consensus S (nearest %.0f km, "
                                        "cap %.0fs); sp_refine retried each second as buffers fill, P-only fallback at the cap"
                                        % (_cd, _scap)); ACTIVE["s_wait_logged"] = True
                                continue               # located stays False -> re-enter + retry sp_refine next second
                            log("  S-WAIT timeout (%.0fs > cap %.0fs): no consensus S arrived -> P-only fallback (event not lost)"
                                % (_evage, _scap))
                        _send_fast(loc, Mfinal, first4, origin, mag_reps, mlabel, bigflag, mdetail)   # S-refined, or P-only (fallback / bigflag)
                    ACTIVE["located"] = True

            # PARTIAL: timed out with >=1 real (SNR-passing) pick but <4
            elif (not ACTIVE["located"] and not ACTIVE["partial"]
                  and (time.time() - ACTIVE["t0_wall"]) > (PARTIAL_TIMEOUT + MAG_LAG)):
                if len(good) >= 3:                       # design decision 2026-06-06: notify ONLY at >=3 picks
                    _send_partial(good)                  # (1-2 picks fire too often on noise -> stay silent)
                ACTIVE["partial"] = True
            # close the event some time after resolution — ALSO when it was only ever rejected (RMS/grid-edge):
            # a rejected noise cluster used to stay open forever (1 Hz re-locate/reject loop, 21k log lines/day)
            # and a LATER REAL event could never open its own cluster until a restart.
            if ACTIVE["t0_wall"] and (time.time() - ACTIVE["t0_wall"]) > (ASSOC_WINDOW + 60):
                _reset_event()

def hub_guard():
    """Keep the hub alive: an unexpected exception must never silently kill the alert thread.
    Event state (REPORTS/ACTIVE) is module-global, so re-entering hub_loop resumes cleanly."""
    while True:
        try:
            hub_loop()
        except Exception:
            log("HUB ERROR (recovered, hub restarting):\n" + traceback.format_exc())
            time.sleep(2.0)

def _reset_event():
    keep = [r for r in REPORTS if ACTIVE["first_P"] is None or abs(r["P_utc"] - ACTIVE["first_P"]) > ASSOC_WINDOW]
    REPORTS[:] = keep
    ACTIVE.update(rescue_tried=False, rescue=None, subset_last=0.0, s_wait_logged=False,  # per-cluster state must NEVER survive a close
                  open=bool(keep), t0_wall=time.time() if keep else None,
                  first_P=keep[0]["P_utc"] if keep else None, located=False, partial=False)
    _SUBSET_RESULT[0] = None    # a search result from the closed cluster must not apply to the next one
    _SUBSET_GEN[0] += 1         # invalidate any in-flight worker spawned for the cluster just closed (review)

# ------------------------------------------------------------------ Step H: parallel far-event associator
# (2026-06-11, research-validated H4 architecture + H2 physics.) WHY: a far great event's P-moveout spans
# MINUTES across the 50-deg network, so the primary 120-s anchor window FRAGMENTS its picks into clusters
# that can't each reach 4 consistent stations — the real Mindanao M7.8 (2026-06-07) was lost exactly this
# way. This SECOND associator admits a pick iff its differential P time vs EVERY current member is
# physically explainable by SOME source in the alert grid (precomputed pairwise moveout envelope, depths
# 10-600 km, +- FAR_SLACK). The PRIMARY path is untouched; the far track runs in its own 1-Hz thread,
# always yields precedence to the primary, reuses the identical locate/D-2-rescue/gate chain, and its
# alerts are deduped against the primary inside _send_fast. Replay numbers (646 h live picks + 275 paired
# trials): Mindanao alerts at ~OT+6 min (56 km err) where live stayed silent; in-network behaviour
# bit-identical; ZERO added false alerts at live pick rates. Bonus: the physics bound auto-rejects
# S-phase/coda picks (BKB +118 s, TNTI +623 s in the replay) that any fixed window would admit.
FAR_DEPTHS       = (10, 100, 300, 600)  # source depths spanned by the moveout envelope (shallow..deep slab)
FAR_SLACK        = 12.0     # s: pick-error + grid-discreteness allowance on the pairwise bounds
FAR_MIN_DWELL    = 180.0    # s: a far cluster lives at least this long after opening
FAR_JOIN_EXTEND  = 120.0    # s: ...and at least this long after the latest member joined
FAR_MAX_LIFE     = 660.0    # s: hard cap (covers ~50-deg P-moveout + pick latency)
FAR_PRECEDENCE_S = 3.0      # s: an alert-ready far candidate waits this long so the 1-Hz primary decides first
FAR_PRECEDENCE_MAX = 75.0   # s: max precedence wait (primary's silent-gate defer caps at 45 s; + margin)
FAR_BUF_MAX_AGE  = 700.0    # s: physics-rejected picks are held for re-anchor at most this long
FAR_STALE_S      = 900.0    # s: far-track stale-origin cutoff (primary keeps 600; far events are still
                            # actionable at 10-15 min — S/surface waves arrive Thailand 11+ min after OT)
FAR_BMIN = None             # (N,N) pairwise moveout envelope, built+cached by _ensure_tt
FAR_BMAX = None
FAR_SIDX = None             # station -> row index in FAR_BMIN/BMAX (sorted station order)
FAR_ACTIVE = None           # the single far cluster (same single-cluster semantics as the primary hub)
FAR_BUF = []                # physics-rejected picks awaiting re-anchor after the current far cluster closes
RECENT_ALERTS = []          # (origin_epoch, lat, lon, wall, assoc, pickset) of every public alert -> identity dedup.
                            # NOTE: the v2 FAR_PRIM_DECIDED station-name yield was REMOVED (review blockers F5/
                            # H2-1/H5-VERDICTBLIND/H5-FALSEYIELD). It matched primary clusters by station NAME with
                            # no onset-time/location/verdict check and latched cl["located"]=True permanently, so a
                            # stale / gate-rejected / noise primary verdict killed a REAL far great event that merely
                            # shared 2 station names within 600 s — the exact opposite of the far track's mission.
                            # The alerted-duplicate case is already covered by the origin/location dedup below; the
                            # rejected-noise case is covered by the far track's OWN identical gate chain (silent /
                            # SNR / amplitude / relevance / stale / zero-mag) — so the yield was redundant as well.
FAR_CODA_ORIGINS = []       # (origin_epoch, lat, lon, wall) of recent FAR-published origins -> reject their S/coda
S_CODA_TOL       = 45.0     # a pick within [-S_CODA_TOL, +90] s of a published origin's S arrival at that station is
                            # its S/coda, not a new P (review H3-SPHANTOM + independent re-review)

def _is_far_coda(r, now):
    """True iff pick r is an S/coda arrival of a recently FAR-published origin. After a great event publishes, its
    S/surface waves trigger P-pickers across the whole network; v2/v3 admitted them and a pure-S foursome EDT-located
    a phantom ~1000 km off (the original H3-SPHANTOM, confirmed still live by the adversarial re-review). Rejected
    PER-PICK at the source so a coda pick can never anchor, join, OR buffer a far cluster — this closes both the
    FAR_BUF-recycle path and the post-close fresh-cluster path that the same-station guard alone missed."""
    if TT_S is None or not FAR_CODA_ORIGINS:
        return False
    sc = STATION_COORDS.get(r["station"])
    if sc is None:
        return False
    for (o_ep, la, lo, w) in FAR_CODA_ORIGINS:
        if now - w > FAR_STALE_S:
            continue
        d = float(_dist_deg(la, lo, np.array([sc["lat"]]), np.array([sc["lon"]]))[0])
        if d > TT_DIST[-1]:
            continue
        s_arr = o_ep + float(np.interp(d, TT_DIST, TT_S))
        if -S_CODA_TOL <= (r["P_utc"] - s_arr) <= 90.0:    # S onset .. early coda of THIS published origin
            return True
    return False

def _far_surface_ghost(first4, loc, now):
    """Reject a far candidate that is actually the S/SURFACE-wave coda of a recently-published far origin
    (review H3-SPHANTOM, surface-wave path that the per-pick TT_S window in _is_far_coda misses). Body-S is
    blocked per-pick by _is_far_coda; but Rayleigh/Love picks arrive HUNDREDS of s after S (group velocity
    ~2.5-4.6 km/s), slip that window, fit the wide P-moveout envelope, and EDT-locate a phantom ~1000-1700 km
    off (independent re-review, replay-proven). A candidate is a ghost iff it locates >250 km from a recent published
    far origin AND >=3 of its 4 locating picks fit THAT origin's surface-wave moveout. Returns the origin or None."""
    if loc is None or not FAR_CODA_ORIGINS:
        return None
    for (o_ep, la, lo, w) in FAR_CODA_ORIGINS:
        if now - w > FAR_STALE_S:                          # age-prune (symmetry with _is_far_coda; review final)
            continue
        if float(_dist_deg(la, lo, np.array([loc["lat"]]), np.array([loc["lon"]]))[0]) * 111.2 <= 250.0:
            continue                                       # genuinely near a published origin -> not a far ghost
        fit = 0; vs = []
        for r in first4:
            sc = STATION_COORDS.get(r["station"])
            if sc is None:
                continue
            dkm = float(_dist_deg(la, lo, np.array([sc["lat"]]), np.array([sc["lon"]]))[0]) * 111.2
            if o_ep + dkm / 4.6 - 35.0 <= r["P_utc"] <= o_ep + dkm / 2.5 + 35.0:    # surface-wave arrival band
                fit += 1
            dt = r["P_utc"] - o_ep
            if dt > 0:
                vs.append(dkm / dt)
        # A true surface coda: >=3/4 picks in the band AND a CONSISTENT implied group velocity from THIS origin
        # (~2.8-4.3 km/s, tight spread). A genuine distinct event's P picks fit its OWN P-moveout, so the surface
        # velocities IMPLIED from this origin scatter -> not flagged. The consistency test cuts the false-positive
        # rate ~26% -> ~6% (final review); and a ghost false-positive only suppresses the redundant far copy
        # — the primary track still publishes the event — so the residual is benign.
        if fit >= 3 and len(vs) >= 4 and 2.8 <= float(np.median(vs)) <= 4.3 and float(np.std(vs)) <= 0.35:
            return (o_ep, la, lo)
    return None

def _far_member_ok(r):
    """Physics membership: dT vs EVERY member must fit the gridded-source moveout envelope (+- slack)."""
    i = FAR_SIDX.get(r["station"]) if FAR_SIDX else None
    if i is None:
        return False
    for m in FAR_ACTIVE["members"]:
        j = FAR_SIDX[m["station"]]
        dt = r["P_utc"] - m["P_utc"]
        if not (FAR_BMIN[j, i] - FAR_SLACK <= dt <= FAR_BMAX[j, i] + FAR_SLACK):
            return False
    return True

def _far_close_at():
    c = FAR_ACTIVE
    return min(max(c["t0_wall"] + FAR_MIN_DWELL, c["last_join"] + FAR_JOIN_EXTEND), c["t0_wall"] + FAR_MAX_LIFE)

def _far_ingest(r, now):
    """Feed one report to the far track (caller holds HUB_LOCK). MEMBERSHIP ONLY — locate/gates/alerts all
    happen in the far 1-Hz tick so the far track can never publish ahead of the primary on the same picks."""
    global FAR_ACTIVE
    if FAR_BMIN is None or FAR_SIDX is None or r["station"] not in FAR_SIDX:
        return
    if _is_far_coda(r, now):            # S/coda of a just-published far event -> never anchors/joins/buffers (review H3)
        return
    if FAR_ACTIVE is None:
        FAR_ACTIVE = {"members": [r], "stas": {r["station"]}, "t0_wall": now, "last_join": now,
                      "located": False, "first4_failed": False, "rescue_tried": False, "rescue": None,
                      "ready": None, "tried": set()}
        return
    cl = FAR_ACTIVE
    if r["station"] in cl["stas"]:
        mine = next((m for m in cl["members"] if m["station"] == r["station"]), None)
        # Re-anchor a genuinely different onset (real P after an earlier noise pick) ONLY while the cluster is
        # still UNLOCATED. Once it has LOCATED, a later same-station arrival is the S/coda/surface wave of THIS
        # event — buffering those let every member's S re-trigger re-ingest at close and form a pure-S foursome
        # that EDT mislocated ~470 km off as a phantom second alert (review H3-SPHANTOM). Drop them.
        if mine is not None and abs(r["P_utc"] - mine["P_utc"]) > 30.0 and not cl["located"]:
            FAR_BUF.append(r)
        return
    if _far_member_ok(r):
        cl["members"].append(r); cl["stas"].add(r["station"]); cl["last_join"] = now
        if len(cl["members"]) >= MIN_STATIONS and not cl["located"]:
            log("  FAR: %s joined (physics-consistent, cluster %d)" % (r["station"], len(cl["members"])))
    else:
        FAR_BUF.append(r)                                # not explainable by any common source -> hold for re-anchor

def _far_close(when):
    global FAR_ACTIVE, FAR_BUF
    cl = FAR_ACTIVE
    if len(cl["members"]) >= MIN_STATIONS:
        log("  FAR: cluster closed (%d members, located=%s)" % (len(cl["members"]), cl["located"]))
    FAR_ACTIVE = None
    # v2 (review H-3): an UNLOCATED cluster's members are RECYCLED together with the buffer — a noise
    # anchor that split a real far event's picks no longer destroys them; minus the anchor (oldest
    # pick_wall, the one that opened the cluster) so the same noise pick cannot re-anchor forever.
    recyc = [] if cl["located"] else sorted(cl["members"], key=lambda r: r.get("pick_wall", 0))[1:]
    buf = sorted(recyc + FAR_BUF, key=lambda r: r.get("pick_wall", 0))
    FAR_BUF = []
    for r in buf:                                        # leftover picks re-anchor a fresh far cluster NOW
        _far_ingest(r, when)

def _far_try(now):
    """Locate + D-2 rescue + full gate chain for the far cluster — an exact mirror of the primary hub_loop
    block, against FAR_ACTIVE state. Differences (all reviewed): a tried-composition cache (the primary may
    re-locate the same first-4 each tick; the far track must not ADD that load), and the PRIMARY-PRECEDENCE
    wait before publishing (the far track must never beat the primary to the same event — see below)."""
    cl = FAR_ACTIVE
    if cl is None or cl["located"]:
        return
    good = sorted([m for m in cl["members"] if m.get("snr", 0) >= MIN_SNR_LOCATE], key=lambda r: r["P_utc"])
    if len(good) < MIN_STATIONS:
        return
    if cl["ready"] is None:
        first4 = good[:4]
        comp = tuple(r["station"] for r in first4)
        if comp not in cl["tried"]:
            cl["tried"].add(comp)
            t0 = first4[0]["P_utc"]
            arrivals = [{"station": r["station"], "relative_time": float(r["P_utc"] - t0)} for r in first4]
            l4 = locate_earthquake(arrivals)
            if l4 and l4["rms"] <= MAX_RMS_LOCATE:
                cl["ready"] = {"loc": l4, "first4": first4, "t0": t0, "wall": now}
                log("  FAR: first-4 locate OK %.2f,%.2f RMS %.2fs — precedence wait before gates"
                    % (l4["lat"], l4["lon"], l4["rms"]))
            else:
                cl["first4_failed"] = True
                if l4:
                    log("  FAR: RMS %.1fs > %.0fs — far first-4 not hypocentre-consistent" % (l4["rms"], MAX_RMS_LOCATE))
        if cl["first4_failed"] and not cl["rescue_tried"] and len(good) >= MIN_STATIONS + 1:
            cl["rescue_tried"] = True
            _resc = _loo_rescue(good)                    # the REAL deployed D-2 (first-5 LOO, bar 1.5, resid>=6)
            if _resc:
                cl["rescue"] = _resc
                log("  FAR D-2 RESCUE CANDIDATE: dropped %s (residual %.1fs), subset RMS %.2fs at %.2f,%.2f — PENDING confirmation"
                    % (_resc["excluded"], _resc["exc_resid"], _resc["loc"]["rms"], _resc["loc"]["lat"], _resc["loc"]["lon"]))
        if cl["ready"] is None and cl.get("rescue"):
            _conf = _rescue_confirmed(cl["rescue"], good)   # the REAL deployed +-4 s independent confirmation
            if _conf:
                rs = cl["rescue"]
                sub = sorted(rs["subset"], key=lambda r: r["P_utc"])
                cl["ready"] = {"loc": rs["loc"], "first4": sub, "t0": rs["t0"], "wall": now, "excluded": rs["excluded"]}
                log("  FAR D-2 RESCUE CONFIRMED by %s (excluded %s barred from magnitude/gates) — precedence wait"
                    % (_conf, rs["excluded"]))
    rd = cl["ready"]
    if rd is None:
        return
    # PRIMARY PRECEDENCE: hold an alert-ready far candidate until the primary had its chance. (a) always
    # >= FAR_PRECEDENCE_S so the 1-Hz primary loop runs at least once after readiness; (b) while the primary
    # cluster is still DECIDING the same event (>=4 good picks sharing >=2 stations with us, not yet located)
    # keep holding, up to FAR_PRECEDENCE_MAX (its silent-gate defer caps at 45 s). If the primary alerts
    # meanwhile, the _send_fast dedup below absorbs our duplicate -> in-network behaviour stays identical.
    wait = now - rd["wall"]
    if wait < FAR_PRECEDENCE_S:
        return
    if wait < FAR_PRECEDENCE_MAX and ACTIVE.get("open") and not ACTIVE.get("located"):
        prim = {r["station"] for r in _cluster() if r.get("snr", 0) >= MIN_SNR_LOCATE}
        if len(prim) >= MIN_STATIONS and len(prim & cl["stas"]) >= 2:
            return
    # (FAR_PRIM_DECIDED station-name yield removed — review F5/H2-1/H5-*: it suppressed real far events. The far
    #  track now relies on the precedence hold above + its own gate chain + the origin/location dedup in _send_fast.)
    loc = rd["loc"]; first4 = rd["first4"]; t0 = rd["t0"]
    _ghost = _far_surface_ghost(first4, loc, now)
    if _ghost:                                          # S/surface-wave coda of a just-published far event (review H3)
        log("  FAR SUPPRESS surface-wave ghost: candidate %.2f,%.2f is the coda of a far alert at %.2f,%.2f (>=3 picks fit surface moveout)"
            % (loc["lat"], loc["lon"], _ghost[1], _ghost[2]))
        cl["located"] = True
        return
    exc = rd.get("excluded")
    ordered = [r for r in good if r["station"] != exc] if exc else good
    origin = UTCDateTime(t0 + loc["origin_time"])
    # ---- from here the chain is the primary's, verbatim (magnitude tiering + the 6 gates) ----
    _cal = [r for r in ordered if r["station"] in MAG_CALIBRATED]
    _cal_clean = [r for r in _cal if r.get("cl", {}).get("mint") and not r.get("cl", {}).get("clipped")]   # complete + not gapped
    _cal_cl = [r for r in _cal if r.get("cl")]
    _any_cl = [r for r in ordered if r.get("cl")]
    mag_reps = _cal_clean[:8] or _cal_cl[:8] or _any_cl[:1] or ordered[:1]   # network MEDIAN over calibrated good-pick (2026-06-15)
    Mfinal, mlabel, bigflag, mdetail = combine_magnitude(mag_reps, loc)
    ok_sil, sratio, ntr, nexp, nlagf = silent_station_check(loc, origin, [r["station"] for r in ordered], Mfinal)
    if (not ok_sil) and nlagf > 0 and (now - rd["wall"]) < 45.0:    # v2: anchor at READINESS (cluster age made
        log("  FAR silent-gate DEFER: %d lagging feed(s) — re-checking" % nlagf)   # this branch dead code - review)
        return
    ok_amp, amad = amplitude_consistency_check(ordered, loc)
    snr_mwp = _mwp_snr10(mag_reps[0]["station"], mag_reps[0]["P_utc"]) if mag_reps else 0.0
    mfresh, mage, mcov = mag_station_fresh(mag_reps[0]["station"], mag_reps[0]["P_utc"]) if mag_reps else (False, 1e9, False)
    if not ok_sil:
        log("  FAR REJECT silent-station: %d/%d (%.0f%% < %.0f%%)" % (ntr, nexp, 100 * sratio, 100 * SILENT_MIN_RATIO))
    elif not mfresh:
        log("  FAR REJECT mag-stale: %s data %.0fs old / covers=%s (need <%.0fs + past P+10) -> offline buffer (KAPI phantom guard)"
            % (mag_reps[0]["station"], mage, mcov, MAG_FRESH_SEC))
    elif snr_mwp < MWP_SNR_MIN:
        log("  FAR REJECT mwp-snr: %s SNR10=%.2f < %.1f" % (mag_reps[0]["station"], snr_mwp, MWP_SNR_MIN))
    elif not ok_amp:
        log("  FAR REJECT amplitude-consistency: MAD %.2f > %.2f" % (amad, AMP_MAD_MAX))
    elif not thailand_relevant(loc, Mfinal):
        log("  FAR SUPPRESS far event: %.0f km from Thailand, M%.1f < %.1f" % (_min_thai_km(loc), Mfinal, FAR_MAG))
    elif float(UTCDateTime() - origin) > FAR_STALE_S:    # v2: 900 s — the far lifecycle (660 s) + precedence
        log("  FAR SUPPRESS stale event: origin %.0f s old" % float(UTCDateTime() - origin))   # wait must fit inside
    elif Mfinal <= 0:
        log("  FAR SUPPRESS zero-magnitude: no station had a complete Mwp window")
    else:
        _send_fast(loc, Mfinal, first4, origin, mag_reps, mlabel, bigflag, mdetail, assoc="far-assoc")
        FAR_CODA_ORIGINS.append((float(origin), float(loc["lat"]), float(loc["lon"]), now))   # arm S/coda rejection (H3)
        del FAR_CODA_ORIGINS[:-8]
    cl["located"] = True

def _far_tick(now):
    """One 1-Hz far-track step (caller holds HUB_LOCK)."""
    global FAR_BUF
    # (primary-adjudication observation removed with FAR_PRIM_DECIDED — review F5/H2-1/H5-*.)
    FAR_BUF = [r for r in FAR_BUF if now - r.get("pick_wall", now) <= FAR_BUF_MAX_AGE]
    while FAR_ACTIVE is not None and now >= _far_close_at():
        _far_close(_far_close_at())
    if FAR_ACTIVE is not None:
        _far_try(now)

def far_loop():
    while True:
        time.sleep(1.0)
        if FAR_BMIN is None:
            continue                                     # tables still building at startup
        with HUB_LOCK:
            _far_tick(time.time())

def far_guard():
    """Crash-proof wrapper, same pattern as hub_guard: the far track must never die silently."""
    while True:
        try:
            far_loop()
        except Exception:
            log("FAR-TRACK ERROR (recovered, far track restarting):\n" + traceback.format_exc())
            time.sleep(2.0)

def _send_fast(loc, Mfinal, locstas, origin, magstas, mlabel="", bigflag=False, mdetail=None, ncluster=None, assoc=""):
    # Real event: terminal one-liner now; the bilingual TH/EN report + 2 plots (from the ONE magnitude
    # station) are built in a background thread so the ~5-min wait for the 6-min plot never blocks the hub.
    magrep = magstas[0] if magstas else (locstas[0] if locstas else None)
    if magrep is None:
        return
    # Step H cross-track dedup — v2 (review blockers H-2 x5): the duplicate check applies to FAR-TRACK
    # alerts ONLY. The primary path is NEVER suppressed (a real foreshock->mainshock doublet within
    # 120 s / 3 deg must always publish — v1 would have silenced the mainshock AND its tsunami arm).
    # Recording happens at the END of a successful send (v1 recorded first, so an exception mid-send
    # turned the far retry into permanent self-suppression — review H-4/H-OPS-3).
    o_ep = float(origin)
    # Cross-track dedup by EVENT IDENTITY (review F1 / H2-DOUBLEPUB + independent re-review of the v2 magnitude gate).
    # The primary and the far associator can assemble the SAME physical event from the SAME picks; that must
    # publish ONCE (one fast alert + one tsunami arm). A genuine foreshock->mainshock doublet has DIFFERENT picks
    # (different P times, even on the same stations) so it is NOT "the same event" and BOTH still publish.
    # Identity = >=2 shared (station, P-second) picks: exact for shared reports, ~impossible to collide across two
    # distinct origins. We deliberately do NOT key on magnitude (great-event MwP spread between the two tracks'
    # single magnitude stations routinely exceeds 0.5 -> a magnitude gate double-publishes AND double-arms tsunami)
    # nor on a loose 120 s/3 deg geographic box (it both over-suppressed real doublets and missed S/surface ghosts).
    mypicks = frozenset((r["station"], round(float(r["P_utc"]))) for r in locstas)
    for (po, pla, plo, pw, passoc, ppicks) in RECENT_ALERTS:
        if len(mypicks & ppicks) >= 2:
            log("  %salert suppressed: same event already published %.0f s ago (%d shared picks)"
                % (assoc + " " if assoc else "", time.time() - pw, len(mypicks & ppicks)))
            return
    n = ncluster if ncluster else len(locstas)
    log("*** FAST ALERT%s *** M%.1f  %.3f,%.3f  depth %.0fkm  RMS %.2fs  %d-stn  mag=%s -> bilingual report + plots"
        % (" [far-assoc]" if assoc else "", Mfinal, loc["lat"], loc["lon"], loc["depth"], loc["rms"], n, magrep["station"]))
    locinfo = [(r["station"], float(r.get("snr10", 0.0))) for r in locstas]   # 4 locating stations + their SNR10 -> provenance line
    _nmag = len([r for r in magstas if r.get("station") in MAG_CALIBRATED]) or len(magstas)   # # calibrated stns in the median
    threading.Thread(target=event_media, args=(loc, Mfinal, magrep, origin, n, bigflag, locinfo, locstas, _nmag), daemon=True).start()
    try:                                                                       # full-Mw recal (+ tsunami arming): isolated daemon, never blocks/breaks the alert above
        if FMT is not None and bigflag and ENABLE_FULLMW:
            threading.Thread(target=FMT.fullmw_update, args=(loc, origin, Mfinal), daemon=True).start()
    except Exception as _e:
        log("fullmw spawn error: %s" % _e)
    try:                                                                       # INSTANT precautionary tsunami arm (MwP>=7.0 offshore Thai-threat; gauges + full-Mw refine) — deduped
        if FMT is not None and ENABLE_TSUNAMI_WATCH:
            FMT.maybe_arm_tsunami_now(loc, Mfinal, origin)
    except Exception as _e:
        log("tsunami instant-arm error: %s" % _e)
    RECENT_ALERTS.append((o_ep, float(loc["lat"]), float(loc["lon"]), time.time(), assoc, mypicks))  # record LAST +
    del RECENT_ALERTS[:-12]                                                                   # track tag + pickset (failed send stays retryable)

def _send_partial(cl):
    if not cl:
        return
    picks = ", ".join(f"{r['station']}(P {UTCDateTime(r['P_utc']).strftime('%H:%M:%S')})" for r in sorted(cl, key=lambda r: r["P_utc"]))
    txt = (f"⚠️ NSTRU IRIS Monitoring — INCOMPLETE EVENT\n"
           f"Only {len(cl)} station(s) reported within {PARTIAL_TIMEOUT:.0f} s (need {MIN_STATIONS} to locate).\n"
           f"Picks: {picks}\n"
           f"No epicenter/magnitude issued.")
    log("*** PARTIAL (MONITOR) ***\n" + txt)
    tg(CH_MONITOR, txt)

# ----------------------------------------------------------------------------- per-station detection + worker
PBUF      = defaultdict(lambda: deque(maxlen=3))
NBUF      = defaultdict(lambda: deque(maxlen=3))
DETCOUNT  = defaultdict(int)
COOLDOWN  = {}                 # sta -> wall time until which we ignore
INFLIGHT  = set()

def station_worker(sta, ref_time):
    """On trigger: STA/LTA pick -> wait for P+13 s -> tp3..10 mags -> submit report."""
    try:
        pick_st = get_z_window(sta, ref_time - PICK_WINDOW_SEC, ref_time)   # Z-only
        if pick_st is None:
            return
        p_arrival, cft = pick_p_arrival(pick_st, ref_time)
        if cft <= 0:
            log(f"  {sta}: pick FAILED (CFT=0) — fallback time NOT submitted (Step C-5)")
            return
        log(f"  {sta}: P-arrival {p_arrival} (CFT={cft:.1f})")
        # wait until buffer holds P + 10 s (Mwp uses 10 s of P) before computing magnitude.
        # v2: the wait scales with the station's record cadence — a 25-100 s BH record cannot arrive
        # inside the old fixed 32 s, and Mwp on a truncated window under-reads (review F4).
        need = p_arrival + 10
        deadline = time.time() + max(32.0, min(120.0, 2.0 * float(PKT_SPAN.get(sta, 3.0)) + 10.0))
        while time.time() < deadline:
            if sta in LAST_DATA and LAST_DATA[sta] >= need + 1:
                break
            time.sleep(0.3)
        covered = sta in LAST_DATA and LAST_DATA[sta] >= need + 1
        ev = get_z_window(sta, p_arrival - 8, need + 3)                     # Z-only
        snr = snr_estimate(ev, p_arrival) if ev is not None else 0.0
        snr10 = _mwp_snr10(sta, p_arrival)        # 10 s SNR = RMS(P..P+10)/RMS(P-10..P), Z, 2-8 Hz
        if snr10 < PICK_SNR_MIN:                   # per-pick gate: a noise pick does NOT count as a P
            log("  %s: pick DROPPED (SNR10 %.2f < %.1f) — not counted toward coincidence" % (sta, snr10, PICK_SNR_MIN))
            return
        cl = classical_measure(sta, p_arrival) if covered else {}   # v2: NO magnitude from a truncated window —
        if not covered:                                             # the pick still serves locate/coincidence
            log("  %s: Mwp window not covered by received data — pick submitted for LOCATE only" % sta)
        if cl.get("clipped"):                      # VISIBLE flag when the railed-channel detector fires
            log("  ⚠ %s: CLIPPED / railed channel (flat-top) — excluded from magnitude-consistency; magnitude = LOWER BOUND" % sta)
        submit_report({"station": sta, "P_utc": float(p_arrival), "pick_wall": time.time(),
                       "mags": {}, "snr": snr, "snr10": snr10, "cl": cl})   # mags={}: E3WS tp dropped (Z-only, no-ML)
    except Exception:
        log(f"  worker {sta} error:\n{traceback.format_exc()}")
    finally:
        INFLIGHT.discard(sta)                     # v2: cooldown is set at TRIGGER time in det_loop

def det_loop():
    log("detection loop started (STA/LTA).")
    while True:
        for sta in list(STATION_COORDS.keys()):
            try:
                last = LAST_DATA.get(sta)
                lw = LAST_PKT_WALL.get(sta)
                span = float(PKT_SPAN.get(sta, 3.0))
                # Step C-1 (2026-06-11): liveness by PACKET ARRIVAL with an allowance scaled to the station's
                # own record cadence. The old "(now - data_end) > 8 s" skip silently excluded every feed with
                # >8 s latency (NPW ~80 s lag: ZERO triggers in 41 h) and every 20-sps BH station between records.
                if last is None or lw is None or (time.time() - lw) > max(30.0, 3.0 * span):
                    continue                                  # stream genuinely dead/stale
                if sta in INFLIGHT or time.time() < COOLDOWN.get(sta, 0):
                    continue
                if time.time() < WARMUP_UNTIL.get(sta, 0):
                    continue                                  # response-deconvolution warm-up after (re)connect
                rescan = sta in WARMUP_RESCAN                 # peek only — consumed AFTER a scan actually runs
                prev_scan = LAST_SCANNED.get(sta)
                if prev_scan is not None and not rescan and float(last - prev_scan) <= 0.0:
                    continue                                  # v2: no NEW data since the last scan (kills the
                                                              # same-P duplicate storm + 10-40x CPU, review F1/F7)
                adv = float(last - prev_scan) if prev_scan is not None else 0.0
                # v2: the fresh span must cover BOTH the record depth (chunky BH) and the data-advance since the
                # last scan (bursty flush feeds) — every new sample is inside the next scan's fresh span.
                fresh_s = max(STALTA_FRESH, min(BUFFER_SEC - STALTA_LTA - 2.0, max(1.5 * span + 2.0, adv + 3.0)))
                if rescan:
                    fresh_s = min(BUFFER_SEC - STALTA_LTA - 2.0, WARMUP_SEC + fresh_s)
                win_s = min(BUFFER_SEC, max(STALTA_WIN, STALTA_LTA + STALTA_STA + fresh_s + 2.0))   # window must hold LTA + the whole fresh span
                zwin = get_z_window(sta, last - win_s, last)
                need_n = int((STALTA_LTA + STALTA_STA + STALTA_FRESH) * SAMPLE_RATE)
                if zwin is None or len(zwin[0].data) < need_n:
                    continue                                  # (rescan credit NOT consumed — retry when data suffices)
                raw = zwin[0].data
                tr = zwin[0].copy(); tr.detrend("demean")
                tr.filter("bandpass", freqmin=DET_BAND[0], freqmax=DET_BAND[1], corners=4, zerophase=True)
                df = tr.stats.sampling_rate
                cft = recursive_sta_lta(tr.data, int(STALTA_STA * df), int(STALTA_LTA * df))
                # v2/v3: BLANK the CFT inside zero-fill AND stuck-value runs, plus a tail after them — the
                # LTA-decay-over-flat edge spike must not trigger or burn the retro-scan credit (C3-EDGETRIG, B-2).
                # Tail scales with run length: 3 s for short glitches, a full LTA (20 s) after long outage holes (A-2).
                quiet = np.abs(raw) < max(1e-9, 1e-3 * (float(np.std(raw)) + 1e-9))
                quiet[1:] |= (np.abs(np.diff(raw)) < 1e-12)                # stuck-value telemetry at ANY level (B-2)
                if quiet.any():
                    qd = np.diff(quiet.astype(np.int8)); starts = list(np.flatnonzero(qd == 1) + 1); ends = list(np.flatnonzero(qd == -1) + 1)
                    if quiet[0]: starts.insert(0, 0)
                    if quiet[-1]: ends.append(len(quiet))
                    for a_, b_ in zip(starts, ends):
                        if (b_ - a_) >= int(0.5 * df):
                            tail = 3.0 if (b_ - a_) < int(30.0 * df) else STALTA_LTA
                            cft[a_: min(len(cft), b_ + int(tail * df))] = 0.0
                LAST_SCANNED[sta] = last                      # this buffer state has been fully scanned
                if rescan:
                    WARMUP_RESCAN.discard(sta)                # consume the credit only now (scan really happened)
                    if time.time() < WARMUP_UNTIL.get(sta, 0):
                        WARMUP_RESCAN.add(sta)                # re-armed mid-scan -> keep the NEW credit
                        continue
                on_thr = STALTA_ON_BY.get(sta, STALTA_ON)                  # per-station ON (noisy sites raised)
                fr_n = max(1, min(len(cft) - 1, int(fresh_s * df)))
                seg = cft[-fr_n:]
                recent = float(np.max(seg))                                # is P energy in the per-station fresh span?
                if recent <= on_thr:
                    continue
                on_off = trigger_onset(cft, on_thr, STALTA_OFF)            # sustained-duration quality gate (spike reject)
                on_off = [(a, b) for a, b in on_off if (b - a) >= MIN_TRIG_DUR * df and b >= len(cft) - fr_n]
                if not on_off:
                    continue
                # v2: anchor at the FIRST onset crossing inside the fresh span — argmax landed on the (stronger)
                # S instead of P on chunky records (39.8 s pick error in review sim C1-SPICK).
                # v3 (review A-3): iterate the onsets — skip only the one matching LAST_TRIG, so a REAL P
                # arriving shortly after a previous trigger is still anchored instead of the scan bailing out.
                t_trig = None
                for a_, b_ in on_off:
                    cand = tr.stats.starttime + max(a_, len(cft) - fr_n) / df
                    if abs(float(cand) - LAST_TRIG.get(sta, 0.0)) >= 20.0:
                        t_trig = cand
                        break
                if t_trig is None:
                    continue                                  # every onset in the span was already triggered
                ref = min(last, t_trig + 5.0)
                LAST_TRIG[sta] = float(t_trig)
                COOLDOWN[sta] = time.time() + EVENT_COOLDOWN  # v2: cooldown anchored at TRIGGER time (not worker end)
                INFLIGHT.add(sta)
                log(f"  {sta}: *** STA/LTA TRIGGER *** cft={recent:.1f} ({tr.stats.channel})" + (" [retro]" if rescan else ""))
                threading.Thread(target=station_worker, args=(sta, ref), daemon=True).start()
            except Exception as e:
                tb = traceback.format_exc().strip().splitlines()
                where = tb[-3].strip() if len(tb) >= 3 else ""
                log(f"  detection {sta} error: {e} | {where}")
        time.sleep(0.2)

# ----------------------------------------------------------------------------- station up/down monitor
STATION_UP = {}

def monitor_loop():
    """Station up/down — TERMINAL ONLY (no Telegram) (design decision 2026-06-06). Prints a per-MINUTE
    online/offline roster to the terminal + an on-change line. Telegram is reserved for REAL events."""
    time.sleep(30)
    while True:
        now = UTCDateTime()
        online, offline = [], []
        for sta in STATION_COORDS:
            up = sta in LAST_DATA and (now - LAST_DATA[sta]) < 120
            (online if up else offline).append(sta)
            if sta not in STATION_UP:
                STATION_UP[sta] = up
            elif up != STATION_UP[sta]:
                STATION_UP[sta] = up
                age = (now - LAST_DATA[sta]) if sta in LAST_DATA else -1
                msg = ("%s.%s back ONLINE" % (STATION_NET.get(sta, ''), sta) if up
                       else "%s.%s OFFLINE (no data %.0fs)" % (STATION_NET.get(sta, ''), sta, age))
                log("  MONITOR (terminal-only): " + msg)
        log("  STATION STATUS [%d online / %d offline]  online=%s  offline=%s"
            % (len(online), len(offline), ",".join(sorted(online)) or "-", ",".join(sorted(offline)) or "-"))
        time.sleep(60)

def heartbeat_loop():
    while True:
        time.sleep(30)
        now = UTCDateTime()
        live = sum(1 for s in STATION_COORDS if s in LAST_DATA and (now - LAST_DATA[s]) < 30)
        log(f"  heartbeat: {live}/{len(STATION_COORDS)} stations live; reports buffered={len(REPORTS)}")

# ----------------------------------------------------------------------------- modes
def start_seedlink():
    for server, (selects, _net) in build_selects().items():
        label = "GEOFON" if "geofon" in server else "EarthScope"
        threading.Thread(target=server_thread, args=(server, selects, label), daemon=True).start()

def mode_probe(seconds):
    build_station_tables()
    log(f"PROBE: connecting SeedLink for {seconds} s ...")
    start_seedlink()
    time.sleep(seconds)
    now = UTCDateTime()
    log("PROBE RESULTS (station: packets, last-sample-age s):")
    for sta in STATION_COORDS:
        age = (now - LAST_DATA[sta]) if sta in LAST_DATA else None
        log(f"  {STATION_NET.get(sta,''):3s}.{sta:5s}  pkts={PKT_COUNT.get(sta,0):4d}  "
            f"age={'%.1f' % age if age is not None else 'NO DATA'}")
    live = sum(1 for s in STATION_COORDS if s in LAST_DATA and (now - LAST_DATA[s]) < 30)
    log(f"PROBE: {live}/{len(STATION_COORDS)} stations delivering live data.")

def mode_test_models():
    build_station_tables()
    load_models()
    miss = [s for s in STATION_COORDS if not os.path.exists(PZFILE[s])]
    log(f"coords={len(STATION_COORDS)}  PZ files present={len(STATION_COORDS)-len(miss)}/{len(STATION_COORDS)}"
        + (f"  MISSING={miss}" if miss else ""))
    if DET_GENERIC is None and not MAG_MODELS:          # E3WS models intentionally skipped (STA/LTA + MwP)
        log("unused detection/magnitude models intentionally not loaded (STA/LTA trigger + MwP magnitude) — not used in the live path.")
        log("test-models OK (STA/LTA + MwP mode; PZ files present)." if not miss else "test-models: PZ files MISSING.")
    else:
        nps = sum(1 for s in DET_BY_STATION if DET_BY_STATION[s] is not DET_GENERIC)
        log(f"DET generic: {'OK' if DET_GENERIC else 'FAIL'}   per-station trained: {nps}   MAG: {len(MAG_MODELS)}/8")
        for sta in STATION_COORDS:
            tag = "generic DET_SLRZ" if DET_BY_STATION.get(sta) is DET_GENERIC else f"model[{DET_ASSIGN.get(sta)}]"
            log(f"    {STATION_NET.get(sta,''):3s}.{sta:5s} -> {tag}")
        log("test-models OK." if DET_GENERIC and len(MAG_MODELS) == 8 and not miss else "test-models had issues.")

def mode_run():
    build_station_tables()
    load_models()
    log(f"Stations: {len(STATION_COORDS)}  |  partial_timeout={PARTIAL_TIMEOUT}s  assoc={ASSOC_WINDOW}s  "
        f"min_stations={MIN_STATIONS}  magnitude=MwP@10s  depth={FIXED_DEPTH}km")
    _ensure_tt(); log("travel-time table ready (%d pts to %.0f deg, pre-built so the FIRST event is not delayed)" % (len(TT_TIME), TT_DIST[-1]))
    log("far-event associator ready (Step H: %dx%d moveout bounds, depths %s km, slack %.0fs; disk-cached)"
        % (len(FAR_SIDX), len(FAR_SIDX), "/".join(str(d) for d in FAR_DEPTHS), FAR_SLACK))
    start_seedlink()
    for fn in (det_loop, hub_guard, far_guard, monitor_loop, heartbeat_loop):   # hub + far track crash-proofed
        threading.Thread(target=fn, daemon=True).start()
    if FMT is not None and ENABLE_TSUNAMI_WATCH:                            # terminal-only DART/tide-gauge online-offline monitor
        try: threading.Thread(target=FMT.gauge_monitor_loop, daemon=True).start()
        except Exception as _e: log("gauge_monitor spawn error: %s" % _e)
    log("regional_eews running. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        log("stopped.")

if __name__ == "__main__":
    # fullmw_tsunami does "import regional_eews" at call time; when this file runs AS __main__ that import
    # would build a SECOND, empty module (STATION_COORDS={}) and silently disable the full-Mw update and
    # the Mwpd>=7.5 tsunami arm. Alias the live instance so every later import gets THIS running module.
    sys.modules["regional_eews"] = sys.modules[__name__]
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-models", action="store_true")
    ap.add_argument("--probe", type=int, default=0, metavar="SECONDS")
    a = ap.parse_args()
    if a.test_models:
        mode_test_models()
    elif a.probe:
        mode_probe(a.probe)
    else:
        mode_run()
