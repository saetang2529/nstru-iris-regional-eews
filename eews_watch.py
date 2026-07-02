#!/usr/bin/env python3
"""
eews_watch.py  --  STANDALONE liveness watcher for the NSTRU IRIS regional EEWS.

SEPARATE program. NEVER touches regional_eews.py or the running process. Its only
interaction with the EEWS is READ-ONLY: it reads the monthly log file (reading a
file cannot affect the writer) and runs pgrep/ps (which send NO signal). It sends
no signals, edits no EEWS file, restarts nothing. Only writes: its state file,
Telegram messages, and a status line to stdout (-> screen on a manual run, -> watch.log under cron).

Liveness: the EEWS writes "heartbeat: N/48 stations live" every ~30 s regardless of
earthquakes, so the log mtime is always fresh when alive. Stale log => down/hung.
Month-rollover safe: reads the NEWEST UNCOMPRESSED eews_*.log (never an .xz archive).

Telegram cadence (to the Monitoring channel only):
  ALIVE pulses at scheduled ICT times -- weekdays 06:00/08:45/12:00/15:30/21:30, weekends 08:45.
    First pulse of the day = richer daily summary.
  OFFLINE 24/7: immediate, then a gentle repeat every ~4 h while down.
  BACK ONLINE once, on recovery.

Terminal/watch.log: EVERY run prints a status readout (checked time, result, events
today public vs no-effect, last alert, next check + next pulse) -- so a manual run shows
you the state, and `tail -f watch.log` lets you watch it like regional_eews.py.

Cron (auto, every 10 min):
  */10 * * * * cd ~/IRIS_REGIONAL_EEWS && python eews_watch.py >> watch.log 2>&1
Manual one-shot check (prints the readout):  cd ~/IRIS_REGIONAL_EEWS && python eews_watch.py
"""
import os, sys, glob, json, time, subprocess, urllib.request, urllib.parse, datetime

PROJECT   = os.path.expanduser("~/IRIS_REGIONAL_EEWS")
BOT       = os.environ.get("EEWS_WATCH_BOT", "PUT-YOUR-BOT-TOKEN-HERE")
CHAT      = os.environ.get("EEWS_WATCH_CHAT", "0")                           # set to your CH_MONITOR id
CH_PUBLIC = "public ch"; CH_NOEFF = "no-effect ch"                            # labels only (for the readout)
STATE     = os.environ.get("EEWS_WATCH_STATE", os.path.join(PROJECT, ".eews_watch_state.json"))
STALE_SEC = int(os.environ.get("EEWS_WATCH_STALE", "300"))                    # log idle > this => down (heartbeat is 30 s)
DOWN_REMIND = int(os.environ.get("EEWS_WATCH_DOWN_REMIND", str(4 * 3600)))    # gentle 4 h repeat while down
PROC_PAT  = os.environ.get("EEWS_WATCH_PROC", "python -u regional_eews.py")
SLOT_WIN  = 15 * 60
WD_SLOTS  = [(6, 0), (8, 45), (12, 0), (15, 30), (21, 30)]                    # Mon-Fri (ICT)
WE_SLOTS  = [(8, 45)]                                                         # Sat, Sun (ICT)

def _utcnow(): return datetime.datetime.utcnow()
def _ict(dt):  return dt + datetime.timedelta(hours=7)
def _fmt(dt):  return "%s ICT · %s UTC" % (_ict(dt).strftime("%H:%M"), dt.strftime("%H:%M"))

def _fmt_uptime(secs):   # scales s -> m -> h -> d -> mo -> y (month=30d, year=365d, approx)
    s = int(secs)
    y,  s = divmod(s, 365 * 86400); mo, s = divmod(s, 30 * 86400)
    d,  s = divmod(s, 86400);       h,  s = divmod(s, 3600); m, sec = divmod(s, 60)
    p = []
    if y: p.append("%dy" % y)
    if mo or p: p.append("%dmo" % mo)
    if d or p:  p.append("%dd" % d)
    if h or p:  p.append("%dh" % h)
    if m or p:  p.append("%dm" % m)
    p.append("%ds" % sec)
    return " ".join(p)

def tg(text):
    try:
        data = urllib.parse.urlencode({"chat_id": CHAT, "text": text, "disable_web_page_preview": "true"}).encode()
        urllib.request.urlopen("https://api.telegram.org/bot%s/sendMessage" % BOT, data=data, timeout=20).read()
        return True
    except Exception as e:
        sys.stderr.write("tg error: %s\n" % e); return False

def newest_log():
    logs = glob.glob(os.path.join(PROJECT, "eews_*.log"))
    return max(logs, key=os.path.getmtime) if logs else None

def _grep(path, pat, fixed=False):
    try:
        cmd = ["grep"] + (["-F"] if fixed else []) + [pat, path]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""

def station_count(path):
    lines = [l for l in _grep(path, "stations live").splitlines() if l.strip()]
    if lines:
        try: return lines[-1].split("stations live")[0].strip().split()[-1]
        except Exception: pass
    return "?/?"

def proc_alive_and_uptime():
    try:
        r = subprocess.run(["pgrep", "-f", PROC_PAT], capture_output=True, text=True, timeout=10)
        pid = r.stdout.split()[0] if (r.returncode == 0 and r.stdout.strip()) else None
        if not pid: return False, ""
        e = subprocess.run(["ps", "-o", "etimes=", "-p", pid], capture_output=True, text=True, timeout=10).stdout.strip()
        return True, (_fmt_uptime(int(e)) if e.isdigit() else "")
    except Exception:
        return False, ""

def counts_today(path):                                  # alerts the EEWS sent today, split by destination
    today = _utcnow().strftime("%Y-%m-%d")
    merged = [l for l in _grep(path, "FAST ALERT (TH+EN merged", fixed=True).splitlines() if ("[" + today) in l]
    nf  = sum(1 for l in merged if "OPERATOR-ONLY" in l)  # not-felt -> no-effect channel
    return len(merged) - nf, nf                           # (public, not_felt)

def last_alert(path):
    hdr = [l for l in _grep(path, "*** FAST ALERT *** M", fixed=True).splitlines() if l.strip()]
    if not hdr: return "none yet"
    h = hdr[-1]; ts = h[1:20] if h.startswith("[") else "?"
    try:
        after = h.split("*** FAST ALERT *** M", 1)[1].split(); mag, loc = after[0], after[1]
    except Exception:
        mag, loc = "?", "?"
    merged = [l for l in _grep(path, "FAST ALERT (TH+EN merged", fixed=True).splitlines() if l.strip()]
    route = (CH_NOEFF + " (not-felt)") if (merged and "OPERATOR-ONLY" in merged[-1]) else (CH_PUBLIC + " (public)")
    return "%s UTC | M%s %s -> %s" % (ts, mag, loc, route)

def next_slot_str(ict):
    for (h, m) in (WE_SLOTS if ict.weekday() >= 5 else WD_SLOTS):
        t = ict.replace(hour=h, minute=m, second=0, microsecond=0)
        if t > ict: return t.strftime("%H:%M ICT")
    tmr = ict + datetime.timedelta(days=1); h, m = (WE_SLOTS if tmr.weekday() >= 5 else WD_SLOTS)[0]
    return tmr.replace(hour=h, minute=m, second=0, microsecond=0).strftime("%H:%M ICT (tomorrow)")

def next_check_str(dt):                                  # next 10-min cron boundary (ASCII, for the readout)
    nc = dt.replace(second=0, microsecond=0) + datetime.timedelta(minutes=(10 - dt.minute % 10))
    return "%s ICT / %s UTC" % (_ict(nc).strftime("%H:%M"), nc.strftime("%H:%M"))

def due_slot(ict, fired):
    for (h, m) in (WE_SLOTS if ict.weekday() >= 5 else WD_SLOTS):
        slot = ict.replace(hour=h, minute=m, second=0, microsecond=0)
        sid = ict.strftime("%Y-%m-%d") + "_%02d%02d" % (h, m)
        if 0 <= (ict - slot).total_seconds() < SLOT_WIN and sid not in fired:
            return sid
    return None

def load_state():
    try: return json.load(open(STATE))
    except Exception: return {"status": "unknown", "fired": [], "last_down": 0, "summary_date": ""}

def save_state(s):
    try: json.dump(s, open(STATE, "w"))
    except Exception: pass

def alive_msg(path, rich):                               # the Telegram pulse text
    cnt = station_count(path)
    if rich:
        ok, up = proc_alive_and_uptime(); pub, nf = counts_today(path)
        return ("\U0001f49a NSTRU IRIS EEWS — daily summary · %s stations%s · today %d->public / %d->not-felt · %s"
                % (cnt, (" · up " + up) if up else "", pub, nf, _fmt(_utcnow())))
    return "\U0001f493 NSTRU IRIS EEWS — alive · %s stations · %s" % (cnt, _fmt(_utcnow()))

def status_block(path, healthy, age):                    # the terminal / watch.log readout (ASCII-only: clean in any terminal)
    nowu = _utcnow(); ict = _ict(nowu)
    ok, up = proc_alive_and_uptime(); pub, nf = counts_today(path)
    head = "ALIVE [OK]" if healthy else "OFFLINE [DOWN]"
    tstr = "%s ICT / %s UTC" % (ict.strftime("%H:%M"), nowu.strftime("%H:%M"))
    return "\n".join([
        "EEWS watcher - checked %s" % tstr,
        "  result      : %s | %s stations | uptime %s | last EEWS log %ds ago" % (head, station_count(path), up or "?", int(age)),
        "  events today : %d -> %s (public,felt) | %d -> %s (not-felt)" % (pub, CH_PUBLIC, nf, CH_NOEFF),
        "  last alert   : %s" % last_alert(path),
        "  next check   : ~%s (auto every 10 min) | next pulse %s" % (next_check_str(nowu), next_slot_str(ict)),
    ])

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "demo":      # preview the Telegram messages, touch no state
        p = newest_log()
        tg(alive_msg(p, True)); tg(alive_msg(p, False))
        last = datetime.datetime.utcfromtimestamp(os.path.getmtime(p)) if p else _utcnow()
        tg("\U0001f534 NSTRU IRIS EEWS — OFFLINE · process gone (log idle 12 min) · last %s\n"
           "restart when you can: cd ~/IRIS_REGIONAL_EEWS && ~/eews_venv/bin/python -u regional_eews.py" % _fmt(last))
        tg("✅ NSTRU IRIS EEWS — back online · %s stations · was down ~2h10m · %s" % (station_count(p), _fmt(_utcnow())))
        print(status_block(p, True, 2)); return

    now = time.time(); st = load_state()
    log = newest_log()
    age = (now - os.path.getmtime(log)) if log else 1e9
    healthy = (log is not None) and (age < STALE_SEC)

    if healthy:
        if st.get("status") == "down":
            tg("✅ NSTRU IRIS EEWS — back online · %s stations · %s" % (station_count(log), _fmt(_utcnow())))
        ict = _ict(_utcnow()); sid = due_slot(ict, st.get("fired", []))
        if sid or st.get("status") == "unknown":
            rich = (st.get("summary_date") != ict.strftime("%Y-%m-%d"))
            tg(alive_msg(log, rich))
            if rich: st["summary_date"] = ict.strftime("%Y-%m-%d")
            if sid: st["fired"] = (st.get("fired", []) + [sid])[-12:]
        st["status"] = "alive"
    else:
        ok, _ = proc_alive_and_uptime()
        reason = ("no log file found" if log is None else
                  ("process gone (log idle %d min)" % int(age // 60)) if not ok else
                  ("log idle %d min (process up but not logging — hung?)" % int(age // 60)))
        last = datetime.datetime.utcfromtimestamp(os.path.getmtime(log)) if log else _utcnow()
        if st.get("status") != "down" or (now - st.get("last_down", 0) >= DOWN_REMIND):
            tg("\U0001f534 NSTRU IRIS EEWS — OFFLINE · %s · last %s\n"
               "restart when you can: cd ~/IRIS_REGIONAL_EEWS && ~/eews_venv/bin/python -u regional_eews.py"
               % (reason, _fmt(last)))
            st["last_down"] = now
        st["status"] = "down"
    save_state(st)
    print(status_block(log, healthy, age))               # always: screen on manual run, watch.log under cron

if __name__ == "__main__":
    main()
