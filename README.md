# NSTRU IRIS Regional Earthquake Early-Warning System (EEWS)

A complete, real-time, **zero-software-cost** earthquake early-warning and tsunami-screening
system for Thailand and the surrounding region — running unattended on a single Raspberry Pi 5
(and equally on **any personal laptop or desktop**). It detects, locates, sizes, and publicly
reports earthquakes from the first-arriving P waves, and screens offshore great earthquakes
for Andaman tsunami threat, using **only freely available data** (open SeedLink streams from
EarthScope/IRIS and GEOFON) and free open-source software.

In a blind replay of 6,356 cataloged earthquakes against the USGS catalog, the deployed
pipeline achieved a magnitude mean absolute error of 0.31, a median epicentral error of 32 km,
and a median origin-time error of −1.9 s.

## The team

| Name | Affiliation |
|---|---|
| Kasemsak Saetang | Program in Physics, Faculty of Education, Nakhon Si Thammarat Rajabhat University, Thailand |
| Wilaiwan Srisawat | Science and Technology Unit, Watthapho Municipal School, Nakhon Si Thammarat, Thailand |
| Pariyakorn Phetkaew | Acoustics and Vibration Group, National Institute of Metrology (Thailand), Pathum Thani, Thailand |
| Sitirug Limpisawad | Department of Mineral Resources, Bangkok, Thailand |
| Amnuay Noypha | Faculty of Education, Nakhon Si Thammarat Rajabhat University, Thailand |
| Helmut Dürrast | Geophysics Research Center, Faculty of Science, Prince of Songkla University, Hat Yai, Thailand |

**Contact:** light2529@gmail.com

**Live demonstration** (the running system's real output):
- Public alert channel (events affecting Thailand): https://t.me/nstru_eews
- No-effect events channel: https://t.me/+MWo4x2MZeY05Njc1

## Why it costs nothing to run

- **Data are free:** all waveforms arrive in real time over open SeedLink servers
  (`rtserve.earthscope.org:18000`, `geofon.gfz.de:18000`) — no agreements, no fees.
- **Software is free:** Python + ObsPy + NumPy/SciPy + PyTorch/SeisBench (all open source).
- **Hardware is tiny:** the live system uses roughly 1.6 GB of RAM and a few percent of one CPU
  core in steady state. A Raspberry Pi 5 runs it 24/7; any ordinary laptop or desktop is more
  than enough. **No large server is needed.**
- **Dissemination is free:** alerts go to Telegram channels (unlimited subscribers, free Bot API).

## Network updates since release

- **2026-07-02 — IN.SHL (Shillong, NE India)** enrolled and per-station calibrated: first station west of the Sagaing fault (NW back-azimuth).
- **2026-07-03 — II.PALK (Sri Lanka, GSN borehole)** enrolled and calibrated: first station west of the Andaman trench — enables sub-180° azimuthal gaps (fast-path alerts) for tsunami-relevant offshore events.
- The validation results in the accompanying manuscript refer to the 49-feed configuration current at the time of writing.

## Quick start

1. Install Python 3.11 and the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create **one Telegram bot** (talk to `@BotFather`) and **three Telegram channels**:
   a public alert channel, an operator/monitoring channel, and a no-effect channel.
   Add the bot as an *administrator* of all three. Put the bot token and the three
   channel ids into `stations_config.py` (step-by-step instructions are in that file).

3. Run it — a plain terminal is enough; no service or autostart is required:

   ```bash
   python -u regional_eews.py
   ```

   The program prints a live log to the terminal. To test without sending any Telegram
   message, run with `EEWS_DRY_RUN=1`. PhaseNet model weights are downloaded automatically
   by SeisBench on first use. (For a permanent installation you *may* wrap it in systemd,
   but a manually started terminal works exactly the same.)

4. To adapt it to another region: edit the station list in `stations_config.py`, and refit the
   per-station magnitude terms for your stations (see `calib/`).

## What is in this repository

| File | Role |
|---|---|
| `regional_eews.py` | Main real-time engine: SeedLink ingest, STA/LTA+AIC detection, 4-station association, EDT grid location, tiered magnitude (mbP / MwP@10s / Mwpd), false-alarm gates, bilingual alerting |
| `stations_config.py` | Station list, trigger/locator parameters, Telegram configuration (template) |
| `sp_refine.py` | PhaseNet 2-of-2 consensus S–P relocation refinement (off the critical path, grid-gated) |
| `fullmw_tsunami.py` | Full-Mw (Mwpd) anti-saturation recompute + two-stage tsunami arm-and-confirm screen against DART gauges |
| `province_effects.py`, `thai_provinces.py`, `province_map.py` | Per-province felt-intensity footer and 300-dpi MMI choropleth map |
| `eews_watch.py` | Independent liveness watchdog (cron) |
| `mm_watch2.py` | Wanted-station-return watcher (cron) |
| `calib/` | Magnitude-calibration scripts (mbP fit + TMD retrospective); MwP/Mwpd ship as fitted JSON coefficient files, with the fitting method described in the paper |
| `responses/`, `*.json` | Instrument responses (StationXML + poles/zeros), fitted calibration coefficients, Thai province polygons |

## Versions (as deployed on the live Raspberry Pi 5)

Python 3.11 · obspy 1.4.2 · numpy 2.3.3 · scipy 1.16.2 · torch 2.5.1 · seisbench 0.11.7 ·
matplotlib 3.10.6 · Pillow 9.4.0 · requests 2.28.1

## If you use this code, please cite

> Saetang, K., Srisawat, W., Phetkaew, P., Limpisawad, S., Noypha, A., & Dürrast, H. (2026).
> *Earthquake early warning and tsunami screening for Thailand on a single Raspberry Pi:
> A zero-software-cost, open-data regional system* [Manuscript submitted for publication].
> (Journal name, volume, and DOI will be added here upon publication.)

A machine-readable citation is provided in `CITATION.cff`.

## License and intended use

Copyright © 2026 Kasemsak Saetang and the NSTRU EEWS team. **Free for non-commercial use
only** — research, education, and non-profit public-benefit earthquake early warning.
Any commercial use is prohibited. See `LICENSE` for the exact terms.

**Safety notice:** this is a research system, not an official warning service.
