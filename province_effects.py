# province_effects.py — felt-effect footer for the EEWS alert. HYBRID layout (2026-06-17).
# build_footer(...) returns (THAI_text, ENGLISH_text) — two MONOLINGUAL blocks (Thai for Thai readers,
# English for tourists), or ('','') if nothing is affected.
#   • felt area spans >= 2 REGIONS  -> REGION SUMMARY: one tidy line per region (great events)
#   • felt area within ONE region   -> per-province CARD grouped by felt level (typical events)
# No seismology jargon (no "S wave"/"surface wave"); colour-circle levels 🔴🟠🟡⚪; ONE time per place =
# when felt shaking arrives (iasp91 S-table), Thai local time (UTC+7) only. Fed the EEWS's OWN Mw, never TMD ML.
# Felt model = Allen-Wald-Worden (2012) Rhypo IPE. <2 ms once per alert.
import math, time
import numpy as np
from thai_provinces import PROVINCES

DEG_KM    = 111.19
U_SURFACE = 3.2
_C0,_C1,_C2,_C4 = 2.085, 1.428, -1.402, 0.078
_M1,_M2 = -0.209, 2.042
FELT_MMI = 3.0
NEAR_KM  = 150.0
BASIN_MMIN, BASIN_DMIN = 5.0, 150.0
BASIN_DMAX = 2000.0   # upper-distance cap (2026-06-17): keeps Syriam 541 / Mandalay 915 / Sumatra 1260-1900 km; cuts teleseismic (Japan 4475)
BASIN_BUMP, BASIN_FELT = 1.5, 2.3   # anchored to the real Syriam M5.2@541km (amplified 2.37) that swayed 50+ BKK towers
SAT_M    = 7.0
CARD_MAXNAMES = 6

REGION_ORDER = {"North": 0, "Central": 1, "Northeast": 2, "South": 3}
_REGION_NAME = {
    "North":     ("ภาคเหนือ",  "Northern Thailand"),
    "Central":   ("ภาคกลาง",   "Central Thailand"),
    "Northeast": ("ภาคอีสาน",  "Northeastern Thailand"),
    "South":     ("ภาคใต้",    "Southern Thailand"),
}
# level -> (rank, circle, thai word, english word, thai full effect, english full effect, thai short, english short)
_LVL = {
    "Strong":   (3, "🔴", "แรง",     "STRONG",   "ยืนทรงตัวลำบาก ของตกหล่น อาจเสียหาย", "hard to stand, things fall, damage possible",
                 "แรงสั่นรุนแรง ของอาจตกหล่น", "strong shaking, things may fall"),
    "Moderate": (2, "🟠", "ปานกลาง", "MODERATE", "ตกใจ ของในบ้านหล่น หน้าต่างสั่น",    "startling, items fall, windows rattle",
                 "สั่นปานกลาง รู้สึกได้ชัด", "moderate shaking, clearly felt"),
    "Light":    (1, "🟡", "เบา",     "LIGHT",    "รู้สึกได้ชัด ของแขวนแกว่งไกว",        "clearly felt, hanging objects swing",
                 "รู้สึกได้เบา ๆ", "lightly felt"),
    "Weak":     (0, "⚪", "เบามาก",  "WEAK",     "รู้สึกได้เล็กน้อย",                  "barely felt",
                 "รู้สึกได้เล็กน้อย", "barely felt"),
}

def mmi(M, Rhyp):
    RM = _M1 + _M2*math.exp(M-5.0)
    v = _C0 + _C1*M + _C2*math.log(math.sqrt(Rhyp*Rhyp + RM*RM))
    return v + (_C4*math.log(Rhyp/50.0) if Rhyp > 50.0 else 0.0)

def _gc_km(la1, lo1, la2, lo2):
    p = math.pi/180.0
    a = math.sin((la2-la1)*p/2)**2 + math.cos(la1*p)*math.cos(la2*p)*math.sin((lo2-lo1)*p/2)**2
    return 2*6371.0*math.asin(min(1.0, math.sqrt(a)))

def _band(I):
    if I >= 6: return "Strong"
    if I >= 4: return "Moderate"
    if I >= 3: return "Light"
    return "Weak"

def _hm(epoch):
    return time.strftime("%H:%M:%S", time.gmtime(epoch + 7*3600))   # minute precision (it's ~approx)

def _compute(M, origin_epoch, ev_lat, ev_lon, depth, TT_DIST, TT_S):
    rows = []; nearest = None; nd = 9e18
    for p in PROVINCES:
        d = _gc_km(ev_lat, ev_lon, p["lat"], p["lon"])
        if d < nd: nd = d; nearest = p
        Rh = math.sqrt(d*d + depth*depth)
        st = origin_epoch + float(np.interp(d/DEG_KM, TT_DIST, TT_S))   # felt-shaking (S) arrival
        rows.append((p, d, Rh, mmi(M, Rh), st))
    ground = [r for r in rows if r[3] >= FELT_MMI or (r[0] is nearest and r[1] <= NEAR_KM)]
    basin  = [(p, d) for (p, d, Rh, I, st) in rows
              if M >= BASIN_MMIN and p["site"] == "deep_basin" and BASIN_DMIN <= d <= BASIN_DMAX and (I + BASIN_BUMP) >= BASIN_FELT]
    return ground, basin

def _render_region(th, en, regions, anchor):
    summ = []
    for reg, items in regions.items():
        best_rank = -1; best_u = "Weak"; strongest = None; smax = -9.0; tmin = 9e18
        for (p, d, Rh, I, st) in items:
            u = _band(I); r = _LVL[u][0]
            if r > best_rank: best_rank, best_u = r, u
            if I > smax: smax, strongest = I, p
            if st < tmin: tmin = st
        summ.append((reg, best_u, strongest, tmin, best_rank))
    summ.sort(key=lambda x: (-x[4], REGION_ORDER[x[0]]))
    for (reg, u, strongest, tmin, rank) in summ:
        circ = _LVL[u][1]; thn, enn = _REGION_NAME[reg]; t = _hm(tmin)
        tl = "%s %s — %s · เริ่ม ~%s น." % (circ, thn, _LVL[u][6], t)
        el = "%s %s — %s · from ~%s"     % (circ, enn, _LVL[u][7], t)
        if anchor and strongest is not None:
            tl += " (แรงสุด %s)" % strongest["th"]; el += " (strongest %s)" % strongest["en"]
        th.append(tl); en.append(el)

def _render_card(items, th, en):
    bylvl = {}
    for (p, d, Rh, I, st) in items:
        bylvl.setdefault(_band(I), []).append((p, st))
    first = True
    for u in ("Strong", "Moderate", "Light", "Weak"):
        if u not in bylvl: continue
        if not first: th.append(""); en.append("")
        first = False
        rank, circ, thw, enw = _LVL[u][0], _LVL[u][1], _LVL[u][2], _LVL[u][3]
        th.append("%s %s — %s" % (circ, thw, _LVL[u][4]))
        en.append("%s %s — %s" % (circ, enw, _LVL[u][5]))
        rows = sorted(bylvl[u], key=lambda x: x[1]); show = rows[:CARD_MAXNAMES]; more = len(rows) - len(show)
        mth = (" (+อีก %d จังหวัด)" % more) if more > 0 else ""
        men = (" (+%d more)" % more) if more > 0 else ""
        t = _hm(min(st for (p, st) in rows))   # ONE arrival window per level group (provinces here arrive ~together)
        th.append("%s ~%s น.%s" % (" · ".join(p["th"] for (p, st) in show), t, mth))
        en.append("%s ~%s%s" % (" · ".join(p["en"] for (p, st) in show), t, men))

def build_footer(M, origin_epoch, ev_lat, ev_lon, depth, TT_DIST, TT_S, anchor=True):
    """Hybrid bilingual footer: (THAI, ENGLISH) monolingual blocks, or ('','') if nothing affected.
    >=2 regions -> region summary; 1 region -> per-province card. M = the EEWS Mw (a lower bound if >=7)."""
    ground, basin = _compute(M, origin_epoch, ev_lat, ev_lon, depth, TT_DIST, TT_S)
    if not ground and not basin:
        return "", ""
    sat = M >= SAT_M
    th = ["📍 พื้นที่ที่คาดว่าจะรู้สึกได้" + (" (อย่างน้อย)" if sat else "") + " · เวลาไทย โดยประมาณ"]
    en = ["📍 Areas likely to feel it"   + (" (at least)" if sat else "") + " · Thailand time, approx"]
    regions = {}
    for g in ground:
        regions.setdefault(g[0]["region"], []).append(g)
    if len(regions) >= 2:
        _render_region(th, en, regions, anchor)
    elif len(regions) == 1:
        _render_card(next(iter(regions.values())), th, en)
    if basin:
        bkk = min(basin, key=lambda x: x[1]); t = _hm(origin_epoch + bkk[1]/U_SURFACE)
        th.append(""); th.append("🏙️ กรุงเทพฯ/ภาคกลาง — ตึกสูงอาจโยกช้า ๆ ~%s น." % t)
        en.append(""); en.append("🏙️ Bangkok & central plain — tall buildings may sway slowly ~%s" % t)
    th.append("ℹ️ ชั้นบนของอาคารสูงอาจรู้สึกได้ไกลกว่านี้ · เวลาโดยประมาณ")
    en.append("ℹ️ Upper floors of tall buildings may feel it farther · times approximate")
    return "\n".join(th), "\n".join(en)
