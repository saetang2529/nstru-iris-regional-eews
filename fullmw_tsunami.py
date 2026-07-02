#!/usr/bin/env python
# fullmw_tsunami.py  —  full-Mw recal (Mwpd) + tsunami-watch for regional_eews.py.
# Called from _send_fast() in an ISOLATED daemon thread. Lazy-imports regional_eews (R) inside functions to avoid a
# circular import, uses its OWN TauPyModel (R.TAUP is not thread-safe), and wraps everything in try/except so a fault
# here can NEVER crash the core EEWS or break the existing fast alert. All public posts go through R.tg / R.tg_doc.
import math, datetime, time, os, threading
import numpy as np
import requests
from scipy.signal import butter, filtfilt

UA = {"User-Agent": "NSTRU-IRIS-EEWS/1.0"}
G = 9.81
MWP_UPDATE_DELTA = 0.25          # post a magnitude UPDATE only if full Mw grew >= this vs MwP@10s
TSUNAMI_MW       = 7.5           # full Mw threshold to arm the tsunami watch (offshore + Thai-threat region)
TSUNAMI_MW_FAST  = 7.0           # INSTANT precautionary arm on the saturating MwP@10s (the gauges + full-Mw then refine)
_TSU_ARMED = set()               # event-origin keys already armed -> arm the watch at most ONCE per event (dedup both paths)
_TSU_LOCK = threading.Lock()
# --- SEQUENCE-CODA GUARD (2026-06-14): a great event's long-period coda contaminates a NEW event's Mwpd integral for
# tens of minutes (measured: the M7.7 inflated an 11-min aftershock's network Mwpd +0.8 -> a false 7.49 that would arm).
# When a watch was ALREADY armed for a NEARBY event recently, the active watch already covers the threat, so SUPPRESS the
# redundant new arm. SAFE BY CONSTRUCTION: only ever suppresses a re-arm DURING an active watch, never the first/only arm
# (no prior arm -> never suppressed). The FAST closest-station MwP@10s alert uses a short 10s window robust to coda and is
# unaffected (M6.7-in-coda fast alert = 6.82, correct). ---
CODA_GUARD_MIN = 30.0            # minutes: an M7+ coda lasts tens of min (measured: still +0.8 at 11 min)
CODA_GUARD_KM  = 500.0           # km: a recent arm within this -> the active watch already covers this region
_RECENT_ARMS   = []              # [(origin_epoch, lat, lon)] of arms actually ISSUED (for the coda guard)
T50EX_GREAT    = 1.3             # far-ring (>1000 km) coda-subtracted duration ratio above this -> great/tsunamigenic flag
COMPUTE_DELAY_S  = 300           # (legacy) old fixed delay; superseded by the adaptive gaps in fullmw_update (2026-06-13)
MWPD_WIN_MIN     = 12.0          # adaptive Mwpd: a station needs >= this many s of P (use-what-arrived, no S-P cap) to count

# Indian-Ocean DART buoys relevant to the Thai Andaman coast (NDBC realtime2 ids) + the coast points.
GAUGES = {"23401": (8.86, 88.56, 2800.0), "23461": (4.99, 90.05, 3000.0)}
COAST  = {"Phuket": (7.88, 98.30), "KhaoLak": (8.65, 98.25), "PhiPhi": (7.74, 98.77)}
DEPTH_COAST = 1500.0; DET = 5.0
NDWC = ("\n⚠️ ระบบอัตโนมัติ ไม่ใช่คำเตือนทางการ 100% — โปรดยืนยันและทำตาม ศูนย์เตือนภัยพิบัติแห่งชาติ (ปภ.) โทร 1860"
        "\n    Automatic system, NOT a 100% official warning — confirm with & follow Thailand's NDWC/DDPM (1860).")
# --- split-language posting (2026-06-17): every secondary public post = TWO monolingual messages —
# all-Thai then all-English — each = content + (NDWC) + time-footer + "research, not official" note LAST.
NDWC_TH = "⚠️ ระบบอัตโนมัติ ไม่ใช่คำเตือนทางการ 100% — โปรดยืนยันและทำตาม ศูนย์เตือนภัยพิบัติแห่งชาติ (ปภ.) โทร 1860"
NDWC_EN = "⚠️ Automatic system, NOT a 100% official warning — confirm with & follow Thailand's NDWC/DDPM (1860)."
NOTE_TH = "ℹ️ NSTRU IRIS EEWS · ระบบวิจัย ไม่ใช่ประกาศทางการ"
NOTE_EN = "ℹ️ NSTRU IRIS EEWS · research, not official"

def _foot2(R, o):
    """(thai_lines, english_lines) time footer — origin (ICT+UTC), elapsed, post (ICT+UTC); one post-clock for both."""
    o = R.UTCDateTime(o); oict = o + 7 * 3600; post = R.UTCDateTime(); pict = post + 7 * 3600
    d_th, d_en = R._fmt_delay(float(post - o))
    utc = o.strftime("%Y-%m-%d %H:%M:%S"); putc = post.strftime("%Y-%m-%d %H:%M:%S")
    th = ["🕑 %d %s %d  %s น. (เวลาไทย) · %s UTC" % (oict.day, R.TH_MON[oict.month], oict.year + 543, oict.strftime("%H:%M:%S"), utc),
          "⏱️ ผ่านไป %s หลังเกิด" % d_th,
          "📤 โพสต์ %s น. (เวลาไทย) · %s UTC" % (pict.strftime("%H:%M:%S"), putc)]
    en = ["🕑 %s (Thailand) · %s UTC" % (oict.strftime("%H:%M:%S"), utc),
          "⏱️ %s after the quake" % d_en,
          "📤 posted %s (Thailand) · %s UTC" % (pict.strftime("%H:%M:%S"), putc)]
    return th, en

def _post2(R, th_core, en_core, o, ndwc=True):
    """Send the all-Thai message then the all-English message to CH_FAST (content + NDWC? + time-footer + note last)."""
    th = "\n".join(th_core) + (("\n" + NDWC_TH) if ndwc else "")
    en = "\n".join(en_core) + (("\n" + NDWC_EN) if ndwc else "")
    if o is not None:
        tfh, tfe = _foot2(R, o); th += "\n\n" + "\n".join(tfh); en += "\n\n" + "\n".join(tfe)
    th += "\n\n" + NOTE_TH; en += "\n\n" + NOTE_EN
    R.tg(R.CH_FAST, th); R.tg(R.CH_FAST, en)

def _provmap(R, loc, o_utc, mag):
    """Hook 2 — render + post the province felt-intensity MAP for the FINAL public magnitude `mag` to CH_FAST.
    Fully guarded: a fault here can NEVER affect the magnitude-update / tsunami-arm path. Posts only when a province
    is felt (render returns the felt count; 0 -> skip, so a not-felt event posts nothing). The draw is serialized on
    R._MPL_LOCK (shared with the station plots), so the font set inside province_map.render (rc_context) cannot bleed
    into a concurrent plot."""
    try:
        if not getattr(R, "EEWS_PROVINCE_MAP", False) or mag is None:
            return
        import province_map
        ep = float(R.UTCDateTime(o_utc).timestamp)
        out = "/tmp/iris_provmap_%d.jpg" % int(ep)
        with R._MPL_LOCK:
            nf = province_map.render(loc, float(mag), ep, loc.get("depth", getattr(R, "FIXED_DEPTH", 10.0)),
                                     R.TT_DIST, R.TT_S, out)
        if nf:
            R.tg_doc(R.CH_FAST, out, "🗺️ ความรุนแรงที่คาดว่าจะรู้สึกได้ รายจังหวัด (MMI) · full Mw %.1f · 300 dpi" % float(mag))
    except Exception as e:
        _log("provmap (Hook2) error: %s" % e)

def _log(msg):
    try:
        import regional_eews as R; R.log("[fullmw_tsunami] " + msg)
    except Exception:
        print("[fullmw_tsunami] " + msg)

def thai_threat(lat, lon): return (88.0 <= lon <= 100.0) and (-10.0 <= lat <= 20.0)

def _near_sea(lat, lon, km=40.0):
    """True if the epicentre is offshore OR within ~km of the sea (an outer-arc island / coast) — so a megathrust whose
    epicentre lands on an Andaman/Nicobar/Sumatra outer island still arms, while DEEP-inland strike-slip faults (Sumatran
    fault, Sagaing) do NOT (>~40 km from any sea). Replaces the point-only offshore gate. Fail-safe True if no land mask."""
    import regional_eews as R
    g = getattr(R, "_GLOBE", None)
    if g is None:
        return True
    d = km / 111.0
    for dla in (-d, 0.0, d):
        for dlo in (-d, 0.0, d):
            try:
                if not g.is_land(float(lat + dla), float(lon + dlo)):
                    return True
            except Exception:
                pass
    return False

def _arm_tsunami_once(loc, mw, o_utc):
    """Arm tsunami_watch at most ONCE per event — whether triggered by the instant MwP path or the full-Mw path.
    SEQUENCE-CODA GUARD: suppress a redundant arm if a watch was already armed for a NEARBY event in the last
    CODA_GUARD_MIN min (the active watch already covers the threat; this NEVER suppresses the first/only arm)."""
    import regional_eews as R
    try:
        oe = float(R.UTCDateTime(o_utc)); key = str(R.UTCDateTime(o_utc))[:19]
        with _TSU_LOCK:
            if key in _TSU_ARMED:
                return
            for (ae, ala, alo) in _RECENT_ARMS:               # coda guard: a recent NEARBY arm -> watch already up -> skip
                dtmin = abs(oe - ae) / 60.0; dkm = gc_km((loc["lat"], loc["lon"]), (ala, alo))
                if dtmin <= CODA_GUARD_MIN and dkm <= CODA_GUARD_KM:
                    _TSU_ARMED.add(key)                       # mark handled so retries don't re-evaluate it
                    _log("sequence-coda guard: watch already armed %.0f min / %.0f km away -> suppressing redundant arm (M~%.1f, %s)"
                         % (dtmin, dkm, mw, key))
                    return
            _TSU_ARMED.add(key)
            if len(_TSU_ARMED) > 300:
                _TSU_ARMED.clear(); _TSU_ARMED.add(key)
            _RECENT_ARMS.append((oe, loc["lat"], loc["lon"]))  # record THIS arm for the guard
            if len(_RECENT_ARMS) > 200:
                del _RECENT_ARMS[:100]
        threading.Thread(target=tsunami_watch, args=(loc, mw, o_utc), daemon=True).start()
        _log("tsunami watch ARMED (M~%.1f, %s)" % (mw, key))
    except Exception as e:
        _log("_arm_tsunami_once error: %s" % e)

def maybe_arm_tsunami_now(loc, mwp, o_utc):
    """At fast-alert time: if MwP@10s >= 7.0 offshore + Thai-threat, arm the tsunami watch INSTANTLY (precaution — MwP
    saturates, so arm low and let the gauges + full-Mw refine / stand down). Deduped with the full-Mw arming."""
    import regional_eews as R
    try:
        if not (_near_sea(loc["lat"], loc["lon"]) and thai_threat(loc["lat"], loc["lon"])):
            return
        mwp = mwp or 0
        if mwp >= TSUNAMI_MW_FAST:                                    # standard instant precautionary arm (offshore M>=7.0)
            _arm_tsunami_once(loc, mwp, o_utc); return
        # B-FIX (2026-06-13 review fix) NEAR-FIELD saturation guard (ARM-1/TSU-3): a NEAR-field great quake (coast <= ~60 min)
        # reading in MwP's saturated band (>=6.7, i.e. a true ~M7.5 that the 10-s P method under-reads) cannot wait for the
        # T+5min Mwpd backstop -> arm a PRECAUTIONARY watch NOW (it stands down on the gauges). FAR-field events in this
        # band are caught in time by the robust Mwpd retry, so NO extra far-field watches are created.
        try:
            s = (loc["lat"], loc["lon"])
            tcoast = min(tt_min(gc_km(s, c), DEPTH_COAST) for c in COAST.values())
            if mwp >= 6.7 and tcoast <= 60.0:
                _arm_tsunami_once(loc, mwp, o_utc)
        except Exception:
            pass
    except Exception as e:
        _log("maybe_arm_tsunami_now error: %s" % e)
def gc_km(a, b):
    (la1, lo1), (la2, lo2) = map(lambda p: (math.radians(p[0]), math.radians(p[1])), (a, b))
    return 6371.0 * math.acos(min(1.0, math.sin(la1)*math.sin(la2) + math.cos(la1)*math.cos(la2)*math.cos(lo2-lo1)))
def tt_min(dist_km, depth_m): return dist_km*1000.0/math.sqrt(G*depth_m)/60.0

# ---------------------------------------------------------------- FULL-Mw (Mwpd)
def recompute_mwpd(loc, o_utc):
    """Network-MEDIAN Mwpd (extended-P, measured T0) over mag-eligible stations. Returns (mw, n, used) or (None,0,[])."""
    import regional_eews as R
    from obspy.taup import TauPyModel
    from obspy.geodetics import locations2degrees, gps2dist_azimuth
    from obspy import Stream
    taup = TauPyModel("iasp91")                                  # own instance (R.TAUP not thread-safe)
    cal = getattr(R, "MWPDCAL", {"global_term": 0.0, "station_term": {}})
    elig = getattr(R, "MWPD_ELIGIBLE", set())
    gterm = cal.get("global_term", 0.0); sterm = cal.get("station_term", {})
    o_utc = R.UTCDateTime(o_utc)
    mws = []; used = []
    for sta, c in list(R.STATION_COORDS.items()):
        if elig and sta not in elig:
            continue
        try:
            d = locations2degrees(loc["lat"], loc["lon"], c["lat"], c["lon"])
            if d > 45:
                continue
            arr = taup.get_travel_times(source_depth_in_km=min((loc.get("depth", 10) or 10), 700),
                                        distance_in_degree=d, phase_list=["P", "Pn", "Pg", "p"])
            if not arr:
                continue
            ref = o_utc + min(a.time for a in arr) + 10.0
            Rkm = gps2dist_azimuth(loc["lat"], loc["lon"], c["lat"], c["lon"])[0] / 1000.0
            net, lc, cz, ig = R._STA_SRC.get(sta, (R.STATION_NET.get(sta, ""), "", "HHZ", False))
            loc_q = "*" if net == "TM" else (lc if lc else "*")
            st = R._fdsn(ig).get_waveforms(net, sta, loc_q, cz, ref - 55, ref + 160)
            if not (st and len(st)):
                continue
            tr = st.select(component="Z").merge(method=1, fill_value=0)[0]
            if R._is_clipped(tr):                          # B-FIX (M3, review 2026-06-13): a railed/clipped station yields a
                continue                                    # spurious Mwpd -> exclude it on RAW counts (before resample) so a
                                                            # single bad station can never even contribute to arming a watch.
            if abs(tr.stats.sampling_rate - R.SAMPLE_RATE) > 0.01:
                tr.resample(R.SAMPLE_RATE)
            dt = tr.stats.delta
            p_utc, cft = R.pick_p_arrival(Stream([tr.slice(ref - R.PICK_WINDOW_SEC, ref)]), ref)
            if cft <= 0:
                continue
            vel = tr.copy().remove_response(inventory=R.INV, output="VEL", pre_filt=(0.005, 0.01, 40, 45))
            hfv = vel.copy().filter("highpass", freq=1.0, corners=4, zerophase=True)
            noi = hfv.slice(p_utc - 40, p_utc - 5).data; sig = hfv.slice(p_utc, p_utc + 15).data
            if len(noi) < 10 or len(sig) < 10:
                continue
            if np.sqrt(np.mean(sig**2)) / (np.sqrt(np.mean(noi**2)) + 1e-12) < 5.0:
                continue
            # ADAPTIVE/EARLY Mwpd (2026-06-13): use whatever P has ARRIVED (up to 150 s) instead of waiting a fixed ~150 s, so
            # a station contributes EARLY. NO S-P cap (review-MAJOR-1): a great event needs the FULL rupture even on near
            # stations, and the displacement moment-integral tolerates some S (as the original fixed-150 s window did). A short
            # early window UNDER-reads -> a conservative LOWER BOUND that GROWS over the retries -> arm fires when it crosses 7.5.
            win = min(150.0, float(tr.stats.endtime - p_utc) - 2.0)  # review-MAJOR-1 (2026-06-13): use what has ARRIVED, up to
            if win < MWPD_WIN_MIN:                                   # 150 s, NO S-P cap -> a great event needs the FULL rupture
                continue                                            # even on near stations; the displacement moment-integral
                                                                    # tolerates some S (as the original fixed-150 s window did).
            e = np.cumsum(hfv.slice(p_utc, p_utc + win).data ** 2)
            if len(e) < 20 or e[-1] <= 0:
                continue
            T0 = float(max(8.0, min((np.searchsorted(e, 0.90 * e[-1])) * dt, win)))
            disp = tr.copy().remove_response(inventory=R.INV, output="DISP", pre_filt=(0.005, 0.01, 40, 45))
            sg = disp.slice(p_utc, p_utc + T0).data
            if len(sg) <= 10:
                continue
            mint = float(np.max(np.abs(np.cumsum(sg - np.mean(sg)) * dt)))
            if mint <= 0:
                continue
            Mo = mint * 4 * np.pi * 2600 * (6000.0 ** 3) * (Rkm * 1000.0)
            term = sterm.get("%s.%s" % (sta, cz[:2]), gterm)
            station_mw = (2.0 / 3.0) * (np.log10(Mo) - 9.1) + 0.2 + term
            if station_mw > 9.2:                           # B-FIX (M3, review): reject a non-physical single-station Mwpd
                continue
            mws.append(station_mw); used.append(sta)
        except Exception:
            continue
    if not mws:
        return None, 0, []
    return float(np.median(mws)), len(mws), used

def t50ex_far(loc, o_utc):
    """FAR-RING (>1000 km) coda-subtracted DURATION discriminant — clip/saturation-PROOF great-event flag for when the
    Mwpd cannot get enough far unclipped stations. Returns (T50Ex_median, n) or (None, n). A great event keeps radiating
    high-frequency energy for tens of s -> ratio > T50EX_GREAT; large/moderate decays fast -> below. FAR ring only: at
    near distances a moderate event's own SURFACE waves fall inside the late window and falsely inflate it (validated)."""
    import regional_eews as R
    from obspy.taup import TauPyModel
    from obspy.geodetics import gps2dist_azimuth
    taup = TauPyModel("iasp91")
    o_utc = R.UTCDateTime(o_utc); ratios = []
    for sta, c in list(R.STATION_COORDS.items()):
        try:
            Rkm = gps2dist_azimuth(loc["lat"], loc["lon"], c["lat"], c["lon"])[0] / 1000.0
            if not (1000.0 <= Rkm <= 2600.0):                # far ring: >1000 km (surface waves separate from the P-train)
                continue                                      # but <2600 km so the P-train late window still has real signal
            arr = taup.get_travel_times(source_depth_in_km=min((loc.get("depth", 10) or 10), 700),
                                        distance_in_degree=Rkm / 111.19, phase_list=["P", "Pn", "p"])
            if not arr:
                continue
            P = o_utc + min(a.time for a in arr)
            net, lc, cz, ig = R._STA_SRC.get(sta, (R.STATION_NET.get(sta, ""), "", "HHZ", False))
            st = R._fdsn(ig).get_waveforms(net, sta, "*" if net == "TM" else (lc if lc else "*"), cz, P - 25, P + 70)
            if not (st and len(st)):
                continue
            tr = st.select(component="Z").merge(method=1, fill_value=0)[0]
            if abs(tr.stats.sampling_rate - R.SAMPLE_RATE) > 0.01:
                tr.resample(R.SAMPLE_RATE)
            hf = tr.detrend("demean").filter("bandpass", freqmin=1.0, freqmax=5.0, corners=4, zerophase=True)
            def pw(a, b):
                seg = hf.slice(P + a, P + b).data
                return float(np.mean(seg ** 2)) if len(seg) > 20 else None
            Apre, A25, A50 = pw(-20, -5), pw(0, 25), pw(50, 60)   # pre-event coda / early / late HF power
            if Apre is None or A25 is None or A50 is None or (A25 - Apre) <= 0:
                continue
            ratios.append(float(np.sqrt(max(0.0, A50 - Apre)) / np.sqrt(max(1e-30, A25 - Apre))))  # coda-subtracted
        except Exception:
            continue
    if len(ratios) < 4:                                      # need a few far stations for a trustworthy median
        return None, len(ratios)
    return float(np.median(ratios)), len(ratios)

def fullmw_update(loc, o_utc, m_mwp):
    """Daemon thread: from ~T+5 min, RETRY the network-median Mwpd several times (late stations + full extended-P windows
    fill in), post the magnitude UPDATE once, and ARM the tsunami watch as soon as a robust saturation-proof Mwpd>=7.5
    appears. B-FIX (2026-06-13 review fix): this was a SINGLE shot whose `n<3` early-return killed the tsunami-arm path
    entirely (ARM-2/TSU-4/B3); now it is multi-attempt, the arm check is n-appropriate, and the magnitude-update path can
    never block the arm. _arm_tsunami_once dedups across retries (and with the instant-MwP path)."""
    import regional_eews as R
    try:
        o_utc = R.UTCDateTime(o_utc)
        armable = (getattr(R, "ENABLE_TSUNAMI_WATCH", False)
                   and _near_sea(loc["lat"], loc["lon"]) and thai_threat(loc["lat"], loc["lon"]))
        posted_mw = None; best_mw = None; best_n = 0; prev_mw = None; prev_n = 0; stable = 0; t50_done = False
        gaps = [70.0, 70.0, 80.0, 100.0, 140.0, 180.0, 240.0]        # ADAPTIVE (2026-06-13): first ~T+70 s then progressive
        # (was a fixed first wait of COMPUTE_DELAY_S=300 s); each station now contributes as soon as its S-P-capped window
        # has arrived (recompute_mwpd), so the saturation-proof Mwpd is available + growing far sooner -> the arm crosses 7.5
        # at ~T+2-3 min for a great event instead of T+5. The estimate is a conservative LOWER bound early, so no false arm.
        for i, gap in enumerate(gaps):
            time.sleep(gap)
            mw, n, used = recompute_mwpd(loc, o_utc)
            if mw is None:
                prev_mw = None; stable = 0                            # review-BLOCKER-2: a skipped attempt resets convergence
                _log("fullmw: attempt %d - no SNR>5 stations yet" % (i + 1)); continue
            if best_mw is None or mw > best_mw:
                best_mw, best_n = mw, n
            _log("fullmw: attempt %d Mwpd=%.2f n=%d vs MwP=%.2f" % (i + 1, mw, n, m_mwp or 0.0))
            # magnitude UPDATE on a trustworthy median (n>=3). review-M2: do NOT return early; post AGAIN if the full Mw
            # keeps GROWING (longer T0 windows + more far stations land) so the public number is not frozen low for a great
            # event whose threat scales steeply with magnitude.
            if n >= 3:
                do_post = ((mw - (m_mwp or 0.0)) >= (MWP_UPDATE_DELTA if mw >= 6.9 else 0.45)) if posted_mw is None \
                          else ((mw - posted_mw) >= 0.3)
                if do_post:
                    sat_th = ["⚠️ ขนาดใหญ่มาก วิธี Mwpd ก็เริ่มอิ่มตัว — อาจยังเป็นค่าต่ำสุด ขนาดจริงอาจสูงกว่านี้"] if mw >= 8.0 else []
                    sat_en = ["⚠️ For a great quake even the full-Mw method saturates — this may still be a lower bound; the true magnitude may be higher."] if mw >= 8.0 else []
                    _post2(R, ["🔄 ปรับปรุงขนาด — full Mw = %.1f" % mw,
                               "(เดิม MwP@10s %.1f — วิธีเร็วประเมินต่ำกว่าจริง)" % (m_mwp or 0.0),
                               "📢 ยึดข้อมูลทางการ กรมอุตุฯ (TMD)"] + sat_th,
                              ["🔄 Magnitude update — full Mw = %.1f" % mw,
                               "(was MwP@10s %.1f — the rapid method under-reads)" % (m_mwp or 0.0),
                               "📢 Rely on the official source (TMD / USGS)"] + sat_en,
                              o_utc, ndwc=False)
                    posted_mw = mw; _log("fullmw: posted update Mw %.1f" % mw)
            # TSUNAMI ARM — review-M1: a STRICT SUPERSET of the original (n>=3 & >=7.5) WITH downward margin so a real M7.5
            # whose saturation-proof median lands 7.3-7.49 still arms; review-M3a: n>=2 MINIMUM (no single station can arm).
            # A precautionary WATCH the gauges + cautious stand-down absorb (it never wrongly says "move to high ground").
            if armable and ((n >= 3 and mw >= 7.3) or (n == 2 and mw >= 7.5)):
                _arm_tsunami_once(loc, mw, o_utc)
            # T50Ex FAR-RING BACKSTOP (2026-06-14): clip/saturation-PROOF great-event flag for when the Mwpd can't get
            # enough far unclipped stations. First ATTEMPTED at i>=2 and RETRIED each later attempt until >=4 far-ring (1000-
            # 2600 km) stations have their P+60 s window — the 2600 km edge lands only ~T+6 min (~i=3), so a single fixed i==2
            # attempt would usually find <4 stations and silently give up. Measured exactly ONCE (t50_done is set ONLY on a
            # real measurement, never on a deferred/empty attempt) and only while the Mwpd has NOT already confirmed a great
            # event (best<7.5). ARMS via the same coda-guarded path (an aftershock in a great-event coda is still suppressed).
            # Additive: can only RAISE the arm, never suppress. Runs in this background thread, so it never blocks the fast alert.
            if armable and not t50_done and i >= 2 and (best_mw is None or best_mw < 7.5):
                t50, nt = t50ex_far(loc, o_utc)
                if t50 is not None:                          # far-ring data has landed -> measure ONCE, then stop retrying
                    t50_done = True
                    _log("fullmw: far-ring T50Ex=%.2f (n=%d) [%s]" % (t50, nt, "GREAT" if t50 > T50EX_GREAT else "not-great"))
                    if t50 > T50EX_GREAT:
                        _arm_tsunami_once(loc, max(best_mw or 7.5, 7.5), o_utc)
                else:                                        # <4 far stns yet (P+60 s not arrived) -> leave t50_done False, retry
                    _log("fullmw: far-ring T50Ex deferred (%d far stns ready, need >=4) - retry next attempt" % nt)
            # convergence stop (2026-06-13): once the median is stable (grew <=0.1 over two consecutive n>=3 attempts) AND is
            # sub-7 (not a still-growing great event), the rupture is captured -> stop early instead of running the full clock.
            # convergence stop (2026-06-13; review-BLOCKER-2 hardened): only end early for a CLEARLY-small NON-armable event,
            # after TWO genuine consecutive stable pairs (both n>=3). A great event's median PLATEAUS in the saturated band
            # (6.3-6.7) before far stations land, so ANY early stop while it could still climb to 7.5 would MISS the arm ->
            # armable events ALWAYS run the full gap schedule (cost = a few cheap recompute calls; the original never under-armed).
            stable = stable + 1 if (prev_mw is not None and prev_n >= 3 and n >= 3 and abs(mw - prev_mw) <= 0.1) else 0
            prev_mw, prev_n = mw, n
            if stable >= 2 and n >= 3 and mw < 6.0 and not armable:
                _log("fullmw: Mwpd converged %.2f (n=%d, non-armable) - stop early" % (mw, n)); break
        # post-loop safety net (review-M3a: n>=2 required, no single-station arm) — deduped; covers any in-loop gap.
        if armable and best_mw is not None and best_n >= 2 and best_mw >= 7.5:
            _arm_tsunami_once(loc, best_mw, o_utc)
        # R2-polish (review 2026-06-13): symmetric MAGNITUDE net — the arm uses best_mw, so make the PUBLIC full-Mw show
        # the network best too (the in-loop 0.3 hysteresis can leave the displayed number up to ~0.3 low for a great event).
        # Post ONE final update if the best n>=3 median rounds higher than what was last posted and clears the first-post gate.
        if best_mw is not None and best_n >= 3 and (posted_mw is None or round(best_mw, 1) > round(posted_mw, 1)) \
                and best_mw - (m_mwp or 0.0) >= (MWP_UPDATE_DELTA if best_mw >= 6.9 else 0.45):
            sat_th = ["⚠️ ขนาดใหญ่มาก วิธี Mwpd ก็เริ่มอิ่มตัว — อาจยังเป็นค่าต่ำสุด ขนาดจริงอาจสูงกว่านี้"] if best_mw >= 8.0 else []
            sat_en = ["⚠️ For a great quake even the full-Mw method saturates — this may still be a lower bound; the true magnitude may be higher."] if best_mw >= 8.0 else []
            _post2(R, ["🔄 ปรับปรุงขนาด — full Mw = %.1f" % best_mw,
                       "(เดิม MwP@10s %.1f — วิธีเร็วประเมินต่ำกว่าจริง)" % (m_mwp or 0.0),
                       "📢 ยึดข้อมูลทางการ กรมอุตุฯ (TMD)"] + sat_th,
                      ["🔄 Magnitude update — full Mw = %.1f" % best_mw,
                       "(was MwP@10s %.1f — the rapid method under-reads)" % (m_mwp or 0.0),
                       "📢 Rely on the official source (TMD / USGS)"] + sat_en,
                      o_utc, ndwc=False)
            _log("fullmw: final best update Mw %.1f" % best_mw)
        # PROVINCE FELT-MAP (Hook 2): ONE map with the FINAL PUBLIC magnitude, once the full-Mw recompute has settled.
        # posted_mw = the last magnitude UPDATE actually shown to the public (None if none posted -> the public number
        # stayed m_mwp, so the map uses m_mwp). Matches the final text; never under-represents a big event with a stale
        # early magnitude. Fully guarded in _provmap; render returns 0 (no post) if nothing is felt.
        if getattr(R, "EEWS_PROVINCE_MAP", False):
            _provmap(R, loc, o_utc, posted_mw if posted_mw is not None else m_mwp)
    except Exception as e:
        _log("fullmw_update error: %s" % e)

# ---------------------------------------------------------------- TSUNAMI watch
def fetch_dart(did):
    try:
        r = requests.get("https://www.ndbc.noaa.gov/data/realtime2/%s.dart" % did, headers=UA, timeout=25)
        ts, hh = [], []
        for ln in r.text.splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            p = ln.split()
            if len(p) < 8:
                continue
            try:
                ts.append(datetime.datetime(int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5])))
                hh.append(float(p[7]))
            except Exception:
                continue
        o = np.argsort(ts); return _qc_dart([ts[i] for i in o], np.array(hh)[o])
    except Exception:
        return [], np.array([])

_QC_VALVE_T = [0.0]              # last safety-valve log time (dedup: a chronically glitchy feed must not spam the log)

def _qc_dart(ts, hh):
    """QC the raw DART series (Step-B 2026-06-10, hardened per the 6-lens adversarial review). The 3 cm confirm
    threshold means ONE telemetry glitch through the high-pass de-tide reads as a 'tsunami' -> false public
    WATCH/WARNING. Drops, in order:
    (1) NDBC missing-value sentinels (9999.000; a water column is always > 0) + FUTURE-dated garbled lines (a bad
        timestamp would otherwise fake both feed-freshness and arrival coverage),
    (2) gross garbage vs the series median (tides are ~1-2 m, never 30 m),
    (3) isolated 1-2 sample spikes > 0.30 m off BOTH the backward AND the forward LINEAR EXTRAPOLATION of their
        neighbours (linear-trend-exact, so the tide's ~0.25 m/sample slope cannot eat the margin; two-sided, so a
        glitch corrupts only ONE side of its clean neighbour and good samples are never collateral-censored).
        Flagged CLUSTERS of >= 3 samples are KEPT — that is a wave train, not a glitch: a real 15-min-mode
        Nyquist-aliased tsunami flags densely and must survive (clusters merge across gaps of < 3 clean samples),
    (4) the NEWEST 2 samples are flagged by the backward extrapolation alone (no forward context exists, and a live
        60 s poller always meets fresh garbage at the tail first) and join the SAME cluster rule — so an arriving
        real wave-front (>= 3 anomalous samples incl. the tail) is kept whole, while 1-2 tail glitches are trimmed.
    SAFETY VALVE: if all of the above would still censor > 10% of the series, censor NOTHING (baseline behaviour).
    Verified: real Mindanao M7.8 detections (8.4/13.8 cm) and quiet-day 23401/23461 unchanged."""
    if len(hh) == 0:
        return ts, hh
    fut = datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
    m = (hh > 0.0) & (hh < 9000.0) & np.array([t <= fut for t in ts], bool)
    ts = [t for t, k in zip(ts, m) if k]; hh = hh[m]
    if len(hh) >= 5:
        m = np.abs(hh - np.median(hh)) < 30.0
        ts = [t for t, k in zip(ts, m) if k]; hh = hh[m]
    ts0, hh0 = ts, hh; n0 = len(hh); dropped = 0
    for _ in range(3):                                          # ITERATED: one pair member can mask the other's
        if len(hh) < 5:                                         # reference in one direction (e.g. G1 ~ 2*G2) and
            break                                               # only fall to the filter once its partner is gone
        flag = np.zeros(len(hh), bool)
        flag[2:-2] = ((np.abs(hh[2:-2] - (2.0 * hh[1:-3] - hh[:-4])) > 0.30) &     # off the backward extrapolation
                      (np.abs(hh[2:-2] - (2.0 * hh[3:-1] - hh[4:])) > 0.30))       # AND off the forward one
        flag[-2] |= abs(hh[-2] - (2.0 * hh[-3] - hh[-4])) > 0.30                   # newest 2: backward-only
        flag[-1] |= abs(hh[-1] - (2.0 * hh[-2] - hh[-3])) > 0.30
        bad = np.flatnonzero(flag); drops = []
        if len(bad):
            for run in np.split(bad, np.flatnonzero(np.diff(bad) > 3) + 1):
                if len(run) <= 2:                               # 1-2 samples = glitch; >= 3 = signal, keep
                    drops.extend(run.tolist())
        if not drops:
            break
        dropped += len(drops)
        if dropped > 0.10 * n0:                                 # SAFETY VALVE: that is signal or a chronically bad
            if time.time() - _QC_VALVE_T[0] > 600:              # feed, not isolated glitches -> censor NOTHING
                _QC_VALVE_T[0] = time.time()
                _log("dart QC: spike filter would drop %.0f%% of samples - skipped, treating as signal" % (100.0 * dropped / n0))
            ts, hh = ts0, hh0
            break
        keep = np.ones(len(hh), bool); keep[drops] = False
        ts = [t for t, k in zip(ts, keep) if k]; hh = hh[keep]
    return ts, hh

def detide_peak_cm(ts, hh, t0, t1):
    sec = np.array([(t - ts[0]).total_seconds() for t in ts]); m = np.array([(t0 <= t <= t1) for t in ts])
    sec, hh = sec[m], hh[m]
    if len(sec) < 30:
        return None
    grid = np.arange(sec.min(), sec.max(), 60.0); h = np.interp(grid, sec, hh)
    b, a = butter(3, (1 / (2 * 3600.0)) / ((1 / 60.0) / 2), btype="high")
    return float(np.max(np.abs(filtfilt(b, a, h - np.median(h))))) * 100.0

def classify_tier(pk, near_source):
    if pk is None or pk < 3:
        return None
    if pk >= 10 and near_source:
        return "WARNING"
    return "WATCH"

def gauge_status_line():
    """One-line DART/tide-gauge online/offline status (by data freshness) for the TERMINAL monitor."""
    parts = []
    for g in GAUGES:
        try:
            ts, hh = fetch_dart(g)
            if not ts:
                parts.append("%s OFFLINE(no-data)" % g); continue
            age = (datetime.datetime.utcnow() - ts[-1]).total_seconds() / 60.0
            st = ("online %.0fm" % age) if age < 30 else (("stale %.1fh" % (age/60)) if age < 360 else ("OFFLINE %.0fh" % (age/60)))
            parts.append("%s %s" % (g, st))
        except Exception:
            parts.append("%s err" % g)
    return "  DART/TIDE-GAUGE STATUS: " + " | ".join(parts)

def gauge_monitor_loop():
    """TERMINAL-ONLY periodic DART/tide-gauge online-offline monitor (mirrors the seismic monitor_loop; NO Telegram).
    DART feeds update ~15 min in standard mode, so a 5-min check is responsive without hammering NDBC."""
    import regional_eews as R
    while True:
        try:
            R.log(gauge_status_line())
        except Exception as e:
            try: R.log("[fullmw_tsunami] gauge_monitor error: %s" % e)
            except Exception: pass
        time.sleep(300)

def tsunami_watch(loc, full_mw, o_utc):
    """Offshore Thai-threat M>=7.5: immediate quake-based WATCH, then MONITOR live DART buoys until ~tcoast+2.5 h
    (CONFIRM-with-plot on the first >=3 cm wave-train), then a CAUTIOUS WATCH STAND-DOWN. NEVER an operational
    'all-clear' (that is NDWC's call); the stand-down is softened/withheld if a gauge is offline or the source is
    near-field, and only after the ~2 h post-arrival buffer (research 2026-06-09). Posts to CH_FAST. Fails safe."""
    import regional_eews as R
    try:
        o_utc = R.UTCDateTime(o_utc)
        s = (loc["lat"], loc["lon"])
        tcoast = min(tt_min(gc_km(s, c), DEPTH_COAST) for c in COAST.values())
        cnear = min(COAST, key=lambda k: gc_km(s, COAST[k]))
        gdet = {g: tt_min(gc_km(s, (la, lo)), dp) + DET for g, (la, lo, dp) in GAUGES.items()}
        gbest = min(gdet, key=gdet.get); lead = tcoast - gdet[gbest]
        nearfield = (lead <= 15.0) or (tcoast <= 60.0)
        if nearfield:
            _post2(R, [f"🌊🔴 เฝ้าระวังสึนามิ — แผ่นดินไหว M~{full_mw:.1f} ใต้ทะเล",
                       f"คลื่นอาจถึงชายฝั่งอันดามัน (~{cnear}) ใน ~{tcoast:.0f} นาที",
                       "⚠️ หากอยู่ชายหาด ขึ้นที่สูงทันที (เร็วกว่าทุ่นจะยืนยัน)"],
                      [f"🌊🔴 TSUNAMI WATCH — M~{full_mw:.1f} undersea quake",
                       f"Waves may reach the Andaman coast (~{cnear}) in ~{tcoast:.0f} min",
                       "⚠️ If on a beach, move to high ground now (faster than gauges can confirm)"],
                      o_utc, ndwc=True)
        else:
            _post2(R, [f"🌊🔴 เฝ้าระวังสึนามิ — แผ่นดินไหว M~{full_mw:.1f} ใต้ทะเล",
                       f"กำลังตรวจทุ่น DART {gbest} (นำชายฝั่ง ~{lead:.0f} นาที) — จะแจ้งซ้ำเมื่อยืนยัน"],
                      [f"🌊🔴 TSUNAMI WATCH — M~{full_mw:.1f} undersea quake",
                       f"Checking DART {gbest} (leads coast ~{lead:.0f} min) — will update on confirmation"],
                      o_utc, ndwc=True)
        _log("tsunami WATCH posted (nearfield=%s lead=%.0f tcoast=%.0f)" % (nearfield, lead, tcoast))
        # monitor EVERY 60 s (fastest confirmation -> maximum lead time) until ~tcoast+150 min; keep watching past the
        # first CONFIRM (the largest wave is rarely the first) with a WATCH->WARNING upgrade. Track the latest de-tide
        # so the closing stand-down can attach the tidal-level plot (evidence, including the no-tsunami case).
        deadline = o_utc + (tcoast + 150.0) * 60.0
        confirmed = False; warned = False; covered = set(); last_plot = None; stale_warned = set()
        while R.UTCDateTime() < deadline:
            for g in sorted(gdet, key=gdet.get):
                if R.UTCDateTime() < o_utc + (gdet[g] - 20.0) * 60.0:         # start ~20 min before the wave could reach this gauge
                    continue
                try:
                    ts, hh = fetch_dart(g)
                    if not ts:
                        continue
                    # Step-B staleness/coverage QC (2026-06-10): a stale feed is still fine for DETECTION (a wave already
                    # in the record is real) but must NOT count as no-tsunami evidence — the quiet stand-down requires the
                    # LEADING gauge's record to cover its predicted arrival + 2 h of wave-train (the largest wave is
                    # rarely the first; a feed that dies just after arrival proves nothing — Palu 2018).
                    age_min = (datetime.datetime.utcnow() - ts[-1]).total_seconds() / 60.0
                    if age_min > 90.0 and g not in stale_warned:
                        stale_warned.add(g)
                        _log("DART %s feed STALE (%.0f min old) - usable for detection, NOT for the no-tsunami stand-down" % (g, age_min))
                    if (ts[-1] - o_utc.datetime).total_seconds() / 60.0 > gdet[g] + 120.0:
                        covered.add(g)
                    # A-FIX (2026-06-13 review fix): gate the de-tide PEAK to AFTER the physically-possible tsunami arrival at
                    # THIS gauge — never the pre-origin hour, never the seismic Rayleigh-wave transient (which reaches the
                    # DART within minutes, far ahead of any water wave). gdet[g] = tsunami travel-time + DET(5 min); start
                    # ~10 min before the predicted physical arrival, but at LEAST 10 min post-origin so the Rayleigh
                    # transient and any pre-origin anomaly are always excluded. (review A: T1/TSU-1/GM-1/M4)
                    _arr_min = max(10.0, gdet[g] - DET - 10.0)
                    pk = detide_peak_cm(ts, hh, o_utc.datetime + datetime.timedelta(minutes=_arr_min),
                                        o_utc.datetime + datetime.timedelta(hours=10))
                    if pk is not None:
                        last_plot = (g, ts, hh, pk)
                    near = gc_km(s, (GAUGES[g][0], GAUGES[g][1])) < 1500.0
                    tier = classify_tier(pk, near)
                    if tier and (not confirmed or (tier == "WARNING" and not warned)):   # CONFIRM ASAP (once) + a WATCH->WARNING upgrade
                        lvl_th = "เตือนภัย" if tier == "WARNING" else "เฝ้าระวัง"
                        up_th = " (อัปเกรด)" if (confirmed and tier == "WARNING") else ""
                        up_en = " (upgraded)" if (confirmed and tier == "WARNING") else ""
                        eta = tcoast - gdet[g]
                        _post2(R, [f"🌊🟠 ยืนยันสึนามิจากทุ่นวัด — ระดับ{lvl_th}{up_th}",
                                   f"ทุ่น DART {g} วัดคลื่นได้ {pk:.1f} ซม. — คาดถึง ~{cnear} ใน ~{eta:.0f} นาที",
                                   f"🗺️ ตำแหน่งทุ่น: https://www.google.com/maps?q={GAUGES[g][0]:.4f},{GAUGES[g][1]:.4f}"],
                                  [f"🌊🟠 TSUNAMI {tier}{up_en} — gauge-confirmed",
                                   f"DART {g} measured {pk:.1f} cm — est. {cnear} in ~{eta:.0f} min",
                                   f"🗺️ buoy: https://www.google.com/maps?q={GAUGES[g][0]:.4f},{GAUGES[g][1]:.4f}"],
                                  o_utc, ndwc=True)
                        _try_plot(R, g, ts, hh, o_utc, pk, tier)
                        _log("tsunami %s %.1fcm at %s%s" % (tier, pk, g, up_en)); confirmed = True
                        if tier == "WARNING": warned = True
                except Exception as e:
                    _log("tsunami gauge %s error: %s" % (g, e))
            time.sleep(60)                                                   # re-check EVERY MINUTE (as soon as possible)
        _standdown(R, cnear, nearfield, confirmed, gbest in covered, last_plot, o_utc)  # quiet stand-down only if the
        # LEADING gauge (the one the WATCH message named) truly covered arrival+2h; else the cautious branch posts
    except Exception as e:
        _log("tsunami_watch error: %s" % e)

def _standdown(R, cnear, nearfield, confirmed, covered_arrival, last_plot=None, o_utc=None):
    """ONE closing message that stands down the WATCH we opened, WITH the closing tidal-level plot as evidence (incl. the
    no-tsunami case). Informational only — NEVER 'safe to return' (NDWC's authority). Withheld-to-cautious if a gauge was
    offline (Palu 2018) or the source was near-field."""
    try:
        tail_th = ["📌 ค่าน้อยที่ทุ่นไม่ได้แปลว่าชายฝั่งปลอดภัย · ไม่ใช่ประกาศ 'ปลอดภัย/กลับเข้าพื้นที่' อย่างเป็นทางการ — ปภ. เป็นผู้ตัดสิน",
                   "หากรู้สึกสั่นแรง/นานใกล้ชายฝั่ง อย่ารอประกาศ ขึ้นที่สูงทันที"]
        tail_en = ["📌 Small at the buoy ≠ safe at the shore · NOT an official 'all-clear / safe to return' — that is NDWC's call",
                   "If you feel strong or long shaking near the coast, do not wait — move to high ground"]
        if confirmed:
            head_th = "🌊✅ ปิดเหตุการณ์: ทุ่นพบสึนามิก่อนหน้านี้ — ภัยจากแผ่นดินไหวครั้งนี้กำลังผ่านไป (ยกเลิกเฝ้าระวังในช่องนี้)"
            head_en = "🌊✅ Closing update: a tsunami was gauge-detected earlier — the threat from this quake is now passing (WATCH stood down on this channel)"
        elif not covered_arrival:
            head_th = "🌊⚠️ ข้อมูลทุ่น/เกจไม่ครบหรือออฟไลน์ — ยืนยันไม่ได้ว่ามีหรือไม่มีสึนามิ โปรดยึด ปภ. เป็นหลัก (ยังคงเฝ้าระวัง)"
            head_en = "🌊⚠️ Gauge/DART data incomplete or offline — cannot confirm whether a tsunami occurred; defer to NDWC (stay alert)"
        elif nearfield:
            head_th = "🌊⚠️ ยังไม่พบคลื่นสำคัญเพิ่มจากทุ่น แต่เหตุนี้อยู่ใกล้ฝั่ง (คลื่นอาจถึงก่อนทุ่นตรวจจับ) — ไม่ใช่การยืนยันปลอดภัย ติดตาม ปภ."
            head_en = "🌊⚠️ No further significant wave at the buoys, but this was a NEAR-FIELD source (waves can arrive before any gauge) — NOT a safety confirmation; follow NDWC"
        else:
            head_th = "🌊✅ ปิดเหตุการณ์: ทุ่น DART (de-tided) + เกจชายฝั่ง ไม่พบคลื่นสึนามิสำคัญ — ภัยต่อชายฝั่งอันดามันน่าจะผ่านไปแล้ว (ยกเลิกเฝ้าระวังในช่องนี้)"
            head_en = "🌊✅ Closing update: DART (de-tided) + available coastal gauges show no significant tsunami — the Andaman-coast threat appears to have passed (WATCH stood down on this channel)"
        _post2(R, [head_th] + tail_th, [head_en] + tail_en, o_utc, ndwc=True)
        if last_plot is not None and o_utc is not None:                        # closing tidal-level plot — evidence, incl. the no-tsunami flat trace
            try:
                g, ts, hh, pk = last_plot
                lab = ("below 3 cm — no significant tsunami" if (pk is None or pk < 3)
                       else ("%.1f cm — WARNING" % pk if pk >= 10 else "%.1f cm — WATCH" % pk))
                _try_plot(R, g, ts, hh, o_utc, pk, lab)
            except Exception as _e:
                _log("standdown plot error: %s" % _e)
        _log("tsunami stand-down posted (confirmed=%s covered_arrival=%s nearfield=%s)" % (confirmed, covered_arrival, nearfield))
    except Exception as e:
        _log("standdown error: %s" % e)

def _try_plot(R, g, ts, hh, o_utc, pk, tier):
    """Best-effort 300dpi de-tide plot (tidal raw + isolated tsunami) -> CH_FAST. Failure is logged, never fatal."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.dates as mdates
        t0 = o_utc.datetime - datetime.timedelta(hours=1); t1 = o_utc.datetime + datetime.timedelta(hours=8)
        sec = np.array([(t - ts[0]).total_seconds() for t in ts]); m = np.array([(t0 <= t <= t1) for t in ts])
        sec, hw = sec[m], hh[m]
        if len(sec) < 30:
            return
        grid = np.arange(sec.min(), sec.max(), 60.0); h = np.interp(grid, sec, hw)
        b, a = butter(3, (1 / (2 * 3600.0)) / ((1 / 60.0) / 2), btype="high"); iso = filtfilt(b, a, h - np.median(h)) * 100.0
        gtg = [ts[0] + datetime.timedelta(seconds=float(x)) for x in grid]
        path = "/tmp/eews_tsunami_%s.jpg" % g
        lock = getattr(R, "_MPL_LOCK", None)
        ctx = lock if lock is not None else _NullCtx()
        with ctx:
            fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            ax[0].plot(gtg, h, color="navy", lw=0.8); ax[0].set_ylabel("Raw water level (m)")
            ax[0].set_title("DART %s — tide-gauge raw water level (tides + tsunami)" % g)
            ax[1].plot(gtg, iso, color="crimson", lw=0.9, label="isolated tsunami (de-tided)")
            ax[1].axhline(3, color="grey", ls=":", lw=0.9, label="3 cm — detection threshold")
            ax[1].axhline(10, color="darkorange", ls="--", lw=0.9, label="10 cm — WARNING (near-source)")
            ax[1].set_ylabel("Isolated tsunami (cm)"); ax[1].set_xlabel("UTC")
            ax[1].set_title("tides removed -> isolated tsunami  ·  peak %.1f cm -> %s" % (pk, tier))
            ax[1].legend(loc="upper right", fontsize=7, framealpha=0.85)
            for a_ in ax:
                a_.grid(alpha=0.3); a_.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            fig.tight_layout()
            # NSTRU watermark + copyright + logo — same attribution as the 13s/6min event plots (2026-06-09)
            fig.text(0.5, 0.5, "NSTRU IRIS EEWS", ha="center", va="center", fontsize=42, color="gray", alpha=0.07, rotation=28, zorder=0)
            fig.text(0.995, 0.006, "© NSTRU — IRIS Regional EEWS · research (do not reuse without attribution)",
                     ha="right", va="bottom", fontsize=6.5, color="0.45")
            _lg = getattr(R, "_LOGO", None)
            if _lg is not None:
                try:
                    lax = fig.add_axes([0.885, 0.80, 0.06, 0.135], anchor="NW", zorder=10); lax.imshow(_lg); lax.axis("off")  # TOP-RIGHT, same altitude, clear of the (shortened) title
                except Exception:
                    pass
            fig.savefig(path, dpi=300, format="jpg"); plt.close(fig)
        R.tg_doc(R.CH_FAST, path, "DART %s de-tide (300 dpi) — raw tidal level + isolated tsunami %.1f cm" % (g, pk))
    except Exception as e:
        _log("tsunami plot %s error: %s" % (g, e))

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
