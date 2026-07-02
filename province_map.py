#!/usr/bin/env python
# province_map.py — per-province felt-intensity choropleth for the NSTRU IRIS Regional EEWS.
# ADDITIVE + crash-isolated: imported lazily by event_media (Hook 1, M<6.3) and fullmw_tsunami.fullmw_update
# (Hook 2, M>=6.3 final mag). Uses the EEWS's OWN felt model (province_effects._compute / mmi / _band / _LVL)
# fed the SAME magnitude the alert text shows. Renders with matplotlib patches + stdlib json ONLY — NO geopandas/
# GDAL (the Pi venv has neither). Thai font (Garuda) is BUNDLED in the project dir and applied ONLY inside a
# lock-protected rc_context, so the existing station plots' font is never touched.
import os, json, math
import numpy as np
import matplotlib
from matplotlib import font_manager as _fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Polygon as _MplPoly
from matplotlib.lines import Line2D
import province_effects as PE
from thai_provinces import PROVINCES

PROJECT = os.path.dirname(os.path.abspath(__file__))
_GEOJSON = os.path.join(PROJECT, "thailand_provinces.json")
_LOGO_PATH = os.path.join(PROJECT, "nstru_logo.png")
_FONT_REG = os.path.join(PROJECT, "Garuda.ttf")
_FONT_BLD = os.path.join(PROJECT, "Garuda-Bold.ttf")

# bundle the Thai font (additive — just makes "Garuda" resolvable; does NOT change the global default)
_FONT_OK = False
try:
    if os.path.exists(_FONT_REG):
        _fm.fontManager.addfont(_FONT_REG)
        if os.path.exists(_FONT_BLD):
            _fm.fontManager.addfont(_FONT_BLD)
        _FONT_OK = True
except Exception:
    _FONT_OK = False
_FONT_FAMILY = "Garuda" if _FONT_OK else matplotlib.rcParams.get("font.family", ["sans-serif"])

_ANCH = [(1,"#e9eff7"),(2,"#a9d4ea"),(3,"#86e0d4"),(4,"#8fe084"),(5,"#fff04d"),(6,"#ffb43a"),(7,"#ff6a2b"),(8,"#df241d"),(9,"#9d0d12")]
CMAP = LinearSegmentedColormap.from_list("mmi", [((m-1)/8.0, c) for m, c in _ANCH], N=256)
NORM = Normalize(1, 8)
_TH = {b: PE._LVL[b][2] for b in ("Strong", "Moderate", "Light", "Weak")}   # Thai band words straight from the running model

def _ring_centroid(ring):
    x = ring[:, 0]; y = ring[:, 1]; x1 = np.roll(x, -1); y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y; A = cross.sum() / 2.0
    if abs(A) < 1e-12: return float(x.mean()), float(y.mean())
    return float(((x + x1) * cross).sum() / (6 * A)), float(((y + y1) * cross).sum() / (6 * A))

def _load_geojson(path):
    gj = json.load(open(path)); out = []
    for f in gj["features"]:
        g = f["geometry"]; polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        parts = [np.asarray(poly[0], dtype=float) for poly in polys]
        big = max(parts, key=lambda r: abs((r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1]).sum()))  # 2x shoelace area (numpy-version-agnostic; np.cross dropped 2D vectors in numpy 2.4+)
        cx, cy = _ring_centroid(big)
        out.append({"parts": parts, "cx": cx, "cy": cy})
    return out

_GEO = _load_geojson(_GEOJSON)          # load province polygons ONCE at import
try:
    _LOGO = plt.imread(_LOGO_PATH)
except Exception:
    _LOGO = None

def _nearest_prov(clon, clat):
    return min(PROVINCES, key=lambda p: (p["lon"] - clon) ** 2 + (p["lat"] - clat) ** 2)

def render(loc, M, o_utc, depth, TT_DIST, TT_S, out_path, place_th="", tag=""):
    """Draw the per-province felt-intensity map for magnitude M; return the number of felt+basin provinces drawn
    (0 = nothing felt -> caller should NOT post). Raises on a real failure (the caller wraps this in try/except)."""
    import time as _time
    la, lo = float(loc["lat"]), float(loc["lon"]); dep = float(depth); M = float(M)
    ground, basin = PE._compute(M, float(o_utc), la, lo, dep, TT_DIST, TT_S)
    if not ground and not basin:
        return 0                                          # nothing to show -> tell the caller to skip
    felt_en = {p["en"]: I for (p, d, Rh, I, st) in ground}; basin_en = {p["en"] for (p, d) in basin}

    fig = None
    try:
      with matplotlib.rc_context({"font.family": _FONT_FAMILY}):   # font set LOCALLY (lock-held by caller) -> station plots unaffected
        fig = plt.figure(figsize=(9.4, 11.8))
        ax = fig.add_axes([0.07, 0.085, 0.86, 0.79]); ax.set_facecolor("#dcebf7")   # taller/lower: fills the space freed by removing the bottom box
        felt_parts, basin_parts = [], []
        for r in _GEO:
            p = _nearest_prov(r["cx"], r["cy"])
            dd = PE._gc_km(la, lo, p["lat"], p["lon"]); Rh = math.sqrt(dd * dd + dep * dep)
            col = CMAP(NORM(PE.mmi(M, Rh)))
            isf = p["en"] in felt_en; isb = p["en"] in basin_en
            for ring in r["parts"]:
                ax.add_patch(_MplPoly(ring, closed=True, facecolor=col, edgecolor="0.5", linewidth=0.45, zorder=2))
                if isf: felt_parts.append(ring)
                if isb: basin_parts.append(ring)
        for ring in felt_parts:
            ax.add_patch(_MplPoly(ring, closed=True, facecolor="none", edgecolor="#111", linewidth=2.0, zorder=4))
        for ring in basin_parts:
            ax.add_patch(_MplPoly(ring, closed=True, facecolor="none", edgecolor="#1a4f8a", linewidth=1.4, hatch="////", zorder=5))
        ax.plot([lo], [la], marker="*", color="red", markersize=30, markeredgecolor="black", markeredgewidth=1.3, zorder=8)

        if ground:
            sp = max(ground, key=lambda r: r[3])
            ax.annotate("จ.%s\n%s (MMI %.1f)" % (sp[0]["th"], _TH[PE._band(sp[3])], sp[3]), (sp[0]["lon"], sp[0]["lat"]),
                        xytext=(52, 22), textcoords="offset points", fontsize=12, fontweight="bold", color="#5a0000", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#5a0000", alpha=0.93),
                        arrowprops=dict(arrowstyle="->", color="#5a0000", lw=1.2))
        if basin:
            bkk = min(basin, key=lambda x: x[1])[0]
            ax.annotate("แอ่ง กทม./ภาคกลาง\nตึกสูงอาจโยก", (bkk["lon"], bkk["lat"]),
                        xytext=(40, -8), textcoords="offset points", fontsize=10.5, fontweight="bold", color="#10406e", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1a4f8a", alpha=0.93),
                        arrowprops=dict(arrowstyle="->", color="#1a4f8a", lw=1.2))
        ax.set_xlim(min(96.0, lo - 0.8), 106.6); ax.set_ylim(5.3, max(21.4, la + 0.8))
        ax.set_aspect(1.0 / math.cos(math.radians(13)))
        ax.set_xlabel("ลองจิจูด (°E)"); ax.set_ylabel("ละติจูด (°N)"); ax.grid(alpha=0.18, ls=":")

        leg = [Line2D([], [], marker="*", color="red", markersize=14, markeredgecolor="black", linestyle="None",
                      label="จุดศูนย์กลางแผ่นดินไหว  M%.1f" % M)]
        leg += [Patch(fc=CMAP(NORM(y)), ec="0.3", label=lbl) for y, lbl in
                [(2, "เบามาก  (MMI<3)"), (3.5, "เบา  (3-4)"), (5, "ปานกลาง  (4-6)"), (7, "แรง  (6+)")]]
        if basin: leg += [Patch(fc="none", ec="#1a4f8a", hatch="////", label="แอ่ง กทม./กลาง — ตึกสูงโยก")]
        ax.legend(handles=leg, title="ระดับความรุนแรงที่รู้สึกได้", loc="upper left",
                  bbox_to_anchor=(100.6, 11.6), bbox_transform=ax.transData, fontsize=10, title_fontsize=10.5,
                  framealpha=0.93, edgecolor="0.6")

        utc_s = _time.strftime("%Y-%m-%d %H:%M:%S", _time.gmtime(float(o_utc)))
        ict_s = _time.strftime("%H:%M:%S", _time.gmtime(float(o_utc) + 7 * 3600))
        place_line = place_th if place_th else "%.3f°N, %.3f°E" % (la, lo)
        fig.suptitle("NSTRU EEWS — ความรุนแรงที่คาดว่าจะรู้สึกได้ รายจังหวัด (MMI)%s\n"
                     "M%.1f  ·  %s\n%s UTC  (%s ICT)  ·  ลึก %.0f กม.  ·  งานวิจัย ไม่ใช่ประกาศทางการ"
                     % (tag, M, place_line, utc_s, ict_s, dep), fontsize=12, y=0.985)

        # (bottom info-box REMOVED 2026-06-26 — cleaner map + keeps the felt model/method off the public image;
        #  felt areas are still shown visually by the colour fill + the black felt-province outlines + the strongest-province callout)

        if _LOGO is not None:
            lax = fig.add_axes([0.062, 0.875, 0.085, 0.10], anchor="NW", zorder=10); lax.imshow(_LOGO); lax.axis("off")
        fig.text(0.982, 0.045, "© NSTRU — IRIS Regional EEWS · งานวิจัย (ห้ามนำไปใช้โดยไม่อ้างอิง)",
                 ha="right", va="bottom", fontsize=7, color="0.45")   # pulled up close to the lon axis (was 0.010) — tighter bottom
        fig.savefig(out_path, dpi=300, format="jpg")
    finally:
        if fig is not None:
            plt.close(fig)                                # ALWAYS close -> no figure leak even if a draw step raises
    return len(ground) + len(basin)
