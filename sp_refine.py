#!/usr/bin/env python
"""S-P (and optional P) location refinement for the regional EEWS (>=4P + >=1S single-alert design).
Re-picks S (and, if SP_REPICK_P, also P) with a 2-of-2 PhaseNet stead+diting CONSENSUS on the cluster stations
(off the critical path; STA/LTA still triggers), then re-locates with the refined P + consensus S using a robust
Huber grid locate + 1-pass residual-outlier drop (cons+residdrop). Returns {lat,lon,origin,n_s,stations,s_times}
or None (caller keeps the P-only solution = fallback). s_times feeds the green S-pick markers on the alert plots.

Validated offline 2026-06-15: median loc error 83->42 km. Model = PhaseNet stead+diting consensus."""
import os, numpy as np
from obspy.geodetics import gps2dist_azimuth

DEG = 111.19
_MODELS = None                      # lazy ((stead, diting)) ; None until first use, False if unavailable
SPROB_FLOOR = 0.2                   # picker S_threshold
PPROB_FLOOR = 0.3                   # picker P_threshold
CONS_TOL    = 1.0                   # s: 2-of-2 stead/diting S agreement window
PCONS_TOL   = 0.6                   # s: 2-of-2 P agreement window (P is sharper than S)
PWIN        = 4.0                   # s: a re-picked P must be within this of the STA/LTA P (refine, not re-detect)
RESID_THR   = 4.0                   # s: residual-outlier drop threshold
REPICK_P    = os.environ.get("SP_REPICK_P", "1") == "1"   # use PhaseNet consensus P (else keep STA/LTA P) in the locate
GRID_LAT = np.arange(9.0, 27.01, 0.10)
GRID_LON = np.arange(90.0, 104.01, 0.10)
TORCH_THREADS = int(os.environ.get("SP_TORCH_THREADS", "2"))

def _load_models(log=lambda m: None):
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    try:
        import torch, seisbench.models as sbm
        torch.set_num_threads(TORCH_THREADS)
        stead  = sbm.PhaseNet.from_pretrained("stead");  stead.eval()
        diting = sbm.PhaseNet.from_pretrained("diting"); diting.eval()
        _MODELS = (stead, diting)
        log("  sp_refine: PhaseNet stead+diting loaded (P-repick %s, S-P refinement armed)" % ("ON" if REPICK_P else "off"))
    except Exception as e:
        _MODELS = False
        log("  sp_refine: models unavailable (%s) -- refinement disabled, P-only only" % (str(e)[:60]))
    return _MODELS

def _classify_ps(model, st, p_abs, smax_abs):
    """One PhaseNet inference -> (best_P, best_S). best_P = highest-prob P within +-PWIN of the STA/LTA P
    (a refinement, never a new detection); best_S = highest-prob S in (P+1.5, smax)."""
    try:
        out = model.classify(st, P_threshold=PPROB_FLOOR, S_threshold=SPROB_FLOOR)
    except Exception:
        return None, None
    Ps = [(float(p.peak_time), float(p.peak_value)) for p in out.picks
          if p.phase == "P" and abs(float(p.peak_time) - p_abs) <= PWIN]
    Ss = [(float(p.peak_time), float(p.peak_value)) for p in out.picks
          if p.phase == "S" and p_abs + 1.5 < float(p.peak_time) < smax_abs]
    bp = max(Ps, key=lambda x: x[1]) if Ps else None
    bs = max(Ss, key=lambda x: x[1]) if Ss else None
    return bp, bs

def _interp(d, xs, ys):  return float(np.interp(d, xs, ys))
def _gdd(latv, lonv, slat, slon):
    la = np.deg2rad(latv)[:, None]; lo = np.deg2rad(lonv)[None, :]
    s1 = np.deg2rad(slat); s2 = np.deg2rad(slon)
    return np.rad2deg(np.arccos(np.clip(np.sin(s1)*np.sin(la)+np.cos(s1)*np.cos(la)*np.cos(lo-s2), -1, 1)))
def _huber(r, d=2.0):
    a = np.abs(r); return np.where(a <= d, 0.5*r*r, d*(a - 0.5*d))

def _jlocate(P, S, coords, tt_dist, tt_p, tt_s):
    if len(P) < 4: return None
    t0 = min(P.values())
    pr = [(P[s]-t0) - np.interp(_gdd(GRID_LAT, GRID_LON, coords[s]["lat"], coords[s]["lon"]), tt_dist, tt_p) for s in P]
    OT = np.median(np.stack(pr), axis=0)
    C = np.zeros((len(GRID_LAT), len(GRID_LON)))
    for s in P:
        C += _huber((P[s]-t0) - np.interp(_gdd(GRID_LAT, GRID_LON, coords[s]["lat"], coords[s]["lon"]), tt_dist, tt_p) - OT)
    for s in S:
        C += _huber((S[s]-t0) - np.interp(_gdd(GRID_LAT, GRID_LON, coords[s]["lat"], coords[s]["lon"]), tt_dist, tt_s) - OT)
    i, j = np.unravel_index(np.argmin(C), C.shape)
    return {"lat": float(GRID_LAT[i]), "lon": float(GRID_LON[j])}

def _sd(loc, s, coords): return gps2dist_azimuth(loc["lat"], loc["lon"], coords[s]["lat"], coords[s]["lon"])[0]/1000.0

def _resid_drop(P, S, coords, tt_dist, tt_p, tt_s):
    loc = _jlocate(P, S, coords, tt_dist, tt_p, tt_s)
    if loc is None or not S: return loc, {}
    t0 = min(list(P.values()) + list(S.values()))
    ot = np.median([(P[s]-t0) - _interp(_sd(loc, s, coords)/DEG, tt_dist, tt_p) for s in P])
    keep = {s: v for s, v in S.items()
            if abs((v-t0) - ot - _interp(_sd(loc, s, coords)/DEG, tt_dist, tt_s)) <= RESID_THR}
    return _jlocate(P, keep, coords, tt_dist, tt_p, tt_s), keep

def refine(p_picks, ponly_loc, buffers, coords, tt_dist, tt_p, tt_s, log=lambda m: None):
    """>=4P + >=1 consensus-S single-alert refinement. p_picks {sta:P_abs_epoch}; ponly_loc {lat,lon}; buffers {sta:Stream}.
    Returns {lat,lon,origin,n_s,stations,s_times} refined, OR None (caller keeps ponly_loc). Only first-4-received stations.
    PhaseNet consensus re-picks S (always) and P (if SP_REPICK_P); the locate uses refined P (fallback STA/LTA) + consensus S."""
    if _load_models(log) is False or len(p_picks) < 4:
        return None
    # GRID GATE (2026-07-02 audit fix): the refinement grid covers 9-27N/90-104E
    # (the northern mission). For a P-only prelim OUTSIDE it (0.5 deg margin) a refine could only
    # return an edge-clamped point -- red-team proven to survive the 4 s S-residual filter for
    # Sumatra/Andaman sources (55-785 km errors accepted) -> skip, keep the P-only solution.
    if not (GRID_LAT[0] + 0.5 <= ponly_loc["lat"] <= GRID_LAT[-1] - 0.5 and
            GRID_LON[0] + 0.5 <= ponly_loc["lon"] <= GRID_LON[-1] - 0.5):
        log("  sp_refine: prelim %.2f,%.2f outside refine grid -> skip (P-only kept)"
            % (ponly_loc["lat"], ponly_loc["lon"]))
        return None
    from obspy import UTCDateTime
    models = _MODELS
    first4 = sorted(p_picks, key=lambda s: p_picks[s])[:4]
    consS, consP, s_times = {}, {}, {}
    for s in first4:
        if s not in buffers or s not in coords: continue
        dpre = _sd(ponly_loc, s, coords)
        pred_sp = _interp(dpre/DEG, tt_dist, tt_s) - _interp(dpre/DEG, tt_dist, tt_p)
        p_abs = p_picks[s]
        st = buffers[s].slice(UTCDateTime(p_abs) - 30, UTCDateTime(p_abs) + pred_sp + 40)
        if not st or len(st) < 2: continue
        st = st.copy().merge(method=1, fill_value=0)
        smax = p_abs + pred_sp + 20
        p1, s1 = _classify_ps(models[0], st, p_abs, smax)
        p2, s2 = _classify_ps(models[1], st, p_abs, smax)
        if s1 and s2 and abs(s1[0] - s2[0]) <= CONS_TOL:          # 2-of-2 consensus S
            consS[s] = 0.5 * (s1[0] + s2[0]); s_times[s] = consS[s]
        if REPICK_P and p1 and p2 and abs(p1[0] - p2[0]) <= PCONS_TOL:   # 2-of-2 consensus P (within PWIN of STA/LTA)
            consP[s] = 0.5 * (p1[0] + p2[0])
    if not consS:
        return None
    P4 = {s: (consP.get(s, p_picks[s])) for s in first4}          # refined P where consensus exists, else STA/LTA P
    loc, kept = _resid_drop(P4, consS, coords, tt_dist, tt_p, tt_s)
    if loc is None or not kept:
        return None
    t0 = min(P4.values())
    ot = float(np.median([(P4[s] - t0) - _interp(_sd(loc, s, coords)/DEG, tt_dist, tt_p) for s in P4]))
    log("  sp_refine: %dP-repick + %d consensus-S (%s) -> %.2f,%.2f (was %.2f,%.2f)"
        % (len(consP), len(kept), ",".join(kept), loc["lat"], loc["lon"], ponly_loc["lat"], ponly_loc["lon"]))
    return {"lat": loc["lat"], "lon": loc["lon"], "origin": t0 + ot, "n_s": len(kept),
            "stations": list(kept), "s_times": {s: s_times[s] for s in kept},
            "p_times": dict(consP)}    # {sta: refined PhaseNet P} for the stations that reached 2-of-2 consensus -> red P markers
