"""
Station configuration for the regional IRIS/GEOFON E3WS EEWS pipeline.
Read live via SeedLink (one client per server). Separate from the NSTRU daemon network.
"""

SOURCES = {
    "GEOFON": {
        "seedlink": "geofon.gfz.de:18000",   # IP 139.17.228.65 (GFZ Potsdam)
        "network": "GE",
        "location": "",                       # GE uses blank location code
        # MYANMAR ONLY = NPW (Naypyitaw). 2026-06-05: removed all Sumatra/Indonesia GE
        # stations (GSI, BKNI, LHMI, MNAI, PMBI, SMRI, BBJI, UGM) — not in Myanmar.
        "channels": ["HH?"],                  # HH ONLY (removed BH/HN/BN)
        "stations": ["NPW"],                  # NPW=CMG-3ESP Myanmar (HH). Sumatra LHMI/GSI -> GE_S (BH; their HH not live)
    },
    "TM": {
        "seedlink": "rtserve.earthscope.org:18000",   # EarthScope
        "network": "TM",
        "location": "00",
        "channels": ["HH?"],                  # *** HH ONLY (requirement) ***
        # Nearly all Guralp CMG-3T (120 s broadband, velocity); TMDB = CMG-3T/5T hybrid (200 s).
        "stations": ["CMMT", "MHIT", "CMAI", "CRAI", "LOEI", "NAYO", "NONG", "PBKT",
                     "PHRA", "PRAC", "SKLT", "SRDT", "SRIT", "SURA", "TMDB", "UBPT", "PANO", "CHBT"],   # SURA added 2026-06-06 (Surat Thani, HH; QC+MwP-calibrated term -0.156)
    },
    "MM": {
        "seedlink": "rtserve.earthscope.org:18000",   # EarthScope (same SeedLink server as TM)
        "network": "MM",
        "location": "",                       # MM uses blank location code
        "channels": ["HH?"],                  # HH = Trillium 120QA broadband 100 sps (NOT HN/Titan accel 200 sps)
        "stations": ["TGI", "NGU"],           # Myanmar SOURCE-SIDE -> western azimuth for Myanmar events (2026-06-06)
    },
    "IC": {
        "seedlink": "rtserve.earthscope.org:18000",   # EarthScope
        "network": "IC",
        "location": "10",                     # IC.LSA HH is loc '10' (loc '00' = BH only)
        "channels": ["HH?"],                  # LSA HH 100 sps
        "stations": ["LSA"],                  # Lhasa, Tibet — NW azimuth (2026-06-06)
    },
    "MY": {
        "seedlink": "rtserve.earthscope.org:18000",   # EarthScope
        "network": "MY",
        "location": "",
        "channels": ["BH?"],                  # MY streams BH 20 sps live on rtserve (HH not live) -> resampled to 100
        "stations": ["KUM", "IPM", "KOM", "KSM", "SBM", "KKM"],           # Malaysia — southern azimuth (IPM North dead; Z-only uses Z)
    },
    "GE_S": {
        "seedlink": "geofon.gfz.de:18000",    # GEOFON (same server as NPW; build_selects merges)
        "network": "GE",
        "location": "",
        "channels": ["BH?"],                  # LHMI/GSI stream BH 20 sps on GEOFON (HH in metadata but not live) -> resampled to 100
        "stations": ["LHMI", "GSI", "MNAI", "PMBI", "BBJI", "SMRI", "BKB", "JAGI", "TOLI2", "PLAI", "LUWI", "MMRI", "TNTI", "SOEI", "BNDI", "FAKI"],          # Sumatra STS-2/2.5 — SW azimuth (2026-06-06)
    },
    "MS": {
        "seedlink": "rtserve.earthscope.org:18000",   # EarthScope (Singapore/MY border)
        "network": "MS",
        "location": "",
        "channels": ["HH?"],                  # UBIN streams HH 100 sps on rtserve
        "stations": ["UBIN"],                 # Ubin 1.42N (HH) — southern azimuth, added 2026-06-06
    },
    "IN": {
        "seedlink": "rtserve.earthscope.org:18000",   # EarthScope
        "network": "IN", "location": "",
        "channels": ["HH?"],                  # IN.PBA Port Blair, Andaman Is (HH) — megathrust source-side
        "stations": ["PBA", "SHL"],           # SHL = Shillong, NE India, Trillium 240 — NW/Sagaing azimuth (enrolled 2026-07-02 after QC; feed intermittent — tolerated by design)
    },
    "II": {
        "seedlink": "rtserve.earthscope.org:18000",
        "network": "II", "location": "00",
        "channels": ["BH?"],                  # II.KAPI Sulawesi (IDA GSN)
        "stations": ["KAPI", "PALK"],          # PALK = Sri Lanka STS-6 borehole GSN — WEST back-azimuth for the Andaman trench (enrolled 2026-07-03 after QC: 5/6 trench events picked, 0 false triggers/hr, 100% uptime)
    },
    "AU": {
        "seedlink": "rtserve.earthscope.org:18000",
        "network": "AU", "location": "00",
        "channels": ["BH?"],                  # AU.XMI Christmas Island
        "stations": ["XMI"],
    },
    "PS": {
        "seedlink": "rtserve.earthscope.org:18000",
        "network": "PS", "location": "",
        "channels": ["BH?"],                  # PS.JAY Jayapura, Papua
        "stations": ["JAY"],
    },
}

# Event / location parameters (reused from the running geophone/ADXL355 server code)
EVENT = {
    "fixed_depth_km": 10.0,
    "velocity_model": "iasp91",     # production locator = iasp91 (phases P/p/Pn); saetang_thailand optional
    "min_stations": 4,              # 4-station calculation
    "assoc_window_sec": 120,        # group P-picks into one event up to 120 s
                                    # (aperture ~1450 km; real max t4-t1 = 58 s + SeedLink latency)
    "mag_tp_sec": (3, 10),          # tp window range computed per station
    "mag_tp_default": 3,            # tp3 for FAST EEW (regional-calibrated MAE 0.37 vs tp10 0.23; reports ~P+7s)
    "mag_aggregation": "median",    # MEDIAN across stations (min under-reports big events)
}

# Trigger: E3WS deep DETECTION model + STA/LTA P-pick (same method as the NSTRU stations).
TRIGGER = {
    "use_e3ws_det": True,
    "det_threshold": 0.80,          # mean of last 3 P-probs >= 0.80 AND noise < 0.99
    "stalta_confirm": True,         # STA/LTA is the P-PICKER: Z, bandpass 2-10 Hz, nsta 0.5 s, nlta 10 s, ratio 3.0
    # --- multiscale EDT locator, deployed 2026-06-08 (graduated non-convexity: smooth-coarse -> sharp-refine) ---
    "multiscale_locate": True,      # fixes the 0.3-deg coarse grid missing the sharp peak (epi 249->57 km, recall 47->94%)
    "edt_sigma": 4.0,               # pick+model std (s); 4.0 (was 1.0) — at 200-900 km the 1-D MODEL error dominates (NonLinLoc LOCGAU2)
}

# Telegram output — HOW TO SET UP (one bot + three channels):
#   1. Create a bot with @BotFather on Telegram -> you get the bot token.
#   2. Create THREE Telegram channels: (a) PUBLIC alerts, (b) operator monitoring, (c) no-effect events.
#   3. Add your bot as an ADMINISTRATOR of all three channels.
#   4. Post any message in each channel, then open
#      https://api.telegram.org/bot<TOKEN>/getUpdates  and read each channel's numeric chat id
#      (a negative number like -100xxxxxxxxxx).
#      To test without sending, run with the environment variable EEWS_DRY_RUN=1.
#   5. Put the token and the three ids below.
TELEGRAM = {
    "bot_token":  "PUT-YOUR-BOT-TOKEN-HERE",
    "CH_FAST":    0,   # PUBLIC channel — finished bilingual alerts for events relevant to your country
    "CH_MONITOR": 0,   # operator channel — incomplete-event notices + liveness watcher messages
    "CH_NOTFELT": 0,   # no-effect channel — located events with no felt area (anti-cry-wolf reroute)
    "photo_as_document": True,
    "partial_timeout_sec": 90,      # report to CH_MONITOR if <4 stations report within this window
}

# Instrument responses (fetched 2026-06-05 from FDSN: GE.NPW from GEOFON, TM.* from IRIS).
# All are broadband VELOCITY (2 zeros at origin). Pipeline deconvolves to ACCELERATION exactly
# like nstru20 / the geophone RPi server: st_FV(pb_inst=True, pzfile=<per-station velocity pz>).
RESPONSE_STATIONXML = "responses/IRIS_TM_GE_HH_response.xml"
RESPONSE_PZ_DIR     = "responses/pz"     # per-station E3WS velocity SACPZ: <NET>_<STA>_HH.pz (2 zeros at origin)

def all_selectors():
    """Return {server: [NET.STA.LOC.CHA, ...]} SeedLink selectors."""
    out = {}
    for name, s in SOURCES.items():
        sels = []
        for sta in s["stations"]:
            for cha in s["channels"]:
                sels.append(f'{s["network"]}.{sta}.{s["location"]}.{cha}')
        out[s["seedlink"]] = sels
    return out

if __name__ == "__main__":
    import json
    print(json.dumps(all_selectors(), indent=2))
