#!/usr/bin/env python
"""Wanted-station-return watcher v2 (2026-07-02) — Pi-portable rewrite of mm_watch.py.
Alerts (Telegram, operator channel) when a wanted-but-down station returns to the live SeedLink
feeds, or when a brand-new MM station appears. v2 changes vs the 2026-06-06 laptop version:
 - PROJ = this script's own folder (no hard-coded laptop path)
 - live stream lists via obspy SeedLink INFO (no slinktool dependency — the Pi has none)
 - state is saved even when the Telegram send fails (no crash-loop re-alerts)
 - DRY_RUN=1 prints instead of sending (for offline testing)
Run from cron every 30 min: */30 * * * * cd ~/IRIS_REGIONAL_EEWS && ~/eews_venv/bin/python mm_watch2.py >> mm_watch.log 2>&1"""
import re, json, os, sys, datetime

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
import stations_config as CFG

BOT = CFG.TELEGRAM["bot_token"]
WATCH_CH = str(CFG.TELEGRAM.get("CH_MONITOR", 0))   # alerts go to the operator channel
DRY = os.environ.get("DRY_RUN", "0") == "1"
STATE = os.path.join(PROJ, "mm_watch_seen.json")

cfg = set()
for s in CFG.SOURCES.values():
    cfg.update(s["stations"])
# wanted-if-they-return: west Myanmar (bracket Sagaing) + E-Myanmar + Yunnan + NE India
WATCH = {("MM", "SIM"), ("MM", "HKA"), ("MM", "TMU"), ("MM", "KTN"),
         ("IC", "KMI"), ("CB", "TNC"), ("IN", "SHL")}
CH = re.compile(r"^(HH|BH)[ZNE123]$")

def live_streams(server, port=18000):
    """(net,sta) set of stations with a live HH/BH waveform channel, via SeedLink INFO."""
    from obspy.clients.seedlink.basic_client import Client
    out = set()
    try:
        info = Client(server, port, timeout=30).get_info(level="channel", cache=False)
        for row in info:                      # (net, sta, loc, chan) at channel level
            net, sta = row[0], row[1]
            chan = row[3] if len(row) > 3 else ""
            if CH.match(chan or ""):
                out.add((net, sta))
    except Exception:
        # geofon's INFO trips an obspy channel-level parse bug -> fall back to station level
        # (GE stations are all broadband HH/BH, so station presence is an acceptable proxy)
        try:
            for row in Client(server, port, timeout=30).get_info(level="station", cache=False):
                out.add((row[0], row[1]))
        except Exception as e:
            print("INFO failed for %s: %s" % (server, str(e)[:80]))
    return out

live = set()
for srv in ("rtserve.earthscope.org", "geofon.gfz.de"):
    live |= live_streams(srv)
print("%s live HH/BH stations seen: %d" % (datetime.datetime.utcnow().isoformat(), len(live)))

seen = {}
if os.path.exists(STATE):
    try:
        seen = json.load(open(STATE))
        if isinstance(seen, list):            # legacy v1 state (2026-06 laptop era) was a list of keys
            seen = {k: True for k in seen}
        if not isinstance(seen, dict):
            seen = {}
    except Exception:
        seen = {}

alerts = []
for net, sta in sorted(WATCH):
    key = "%s.%s" % (net, sta)
    if (net, sta) in live and not seen.get(key):
        alerts.append("🎉 WANTED station RETURNED: %s — live HH/BH on open SeedLink. Consider QC + MwP calibration + enrol." % key)
        seen[key] = True
    elif (net, sta) not in live:
        seen[key] = False
new_mm = sorted(s for n, s in live if n == "MM" and s not in cfg)
for sta in new_mm:
    key = "MM.%s(new)" % sta
    if not seen.get(key):
        alerts.append("🆕 NEW MM (Myanmar) station on SeedLink not in config: MM.%s — candidate for enrolment." % sta)
        seen[key] = True

json.dump(seen, open(STATE, "w"))            # save state FIRST — a failed send must not re-alert forever
for msg in alerts:
    if DRY:
        print("[DRY_RUN would send]", msg)
    else:
        try:
            import requests
            requests.post("https://api.telegram.org/bot%s/sendMessage" % BOT,
                          data={"chat_id": WATCH_CH, "text": msg}, timeout=30)
        except Exception as e:
            print("TG send failed (state already saved): %s" % str(e)[:80])
if not alerts:
    print("no watchlist changes")
