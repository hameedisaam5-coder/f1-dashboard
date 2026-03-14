"""
Vercel serverless function: GET /api/replay_data
Returns historical race replay as compact per-lap position JSON.
"""

import os
import json
import pickle
import hashlib
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler

import fastf1
import numpy as np

CACHE_DIR   = "/tmp/fastf1_cache"
REPLAY_DIR  = "/tmp/replay_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(REPLAY_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def build_replay_data(year: int, race: str, session_type: str) -> dict:
    cache_key  = hashlib.md5(f"{year}:{race}:{session_type}:v5".encode()).hexdigest()
    cache_file = os.path.join(REPLAY_DIR, f"{cache_key}.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Load session
    sess = fastf1.get_session(year, race, session_type)
    sess.load(telemetry=True, weather=False, messages=False)

    track_x, track_y = [], []
    try:
        fl  = sess.laps.pick_fastest()
        tel = fl.get_telemetry()
        step = max(1, len(tel) // 2000)
        track_x = tel["X"].iloc[::step].tolist()
        track_y = tel["Y"].iloc[::step].tolist()
    except Exception as ex:
        pass

    SAMPLE_INTERVAL = 2.0
    MAX_DURATION_S  = 7200

    driver_frames: dict[float, list] = {}

    for drv_num in sess.drivers:
        try:
            drv_laps = sess.laps.pick_drivers(drv_num)
            if drv_laps.empty:
                continue
            code = drv_laps.iloc[0]["Driver"]

            # get_telemetry() on LAPS block gets all telemetry at once for this driver! Ultra fast.
            all_tel = drv_laps.get_telemetry()
            if all_tel.empty or "X" not in all_tel.columns:
                continue

            t_raw = all_tel["SessionTime"].dt.total_seconds().to_numpy()
            x_raw = all_tel["X"].to_numpy(dtype=float)
            y_raw = all_tel["Y"].to_numpy(dtype=float)
            s_raw = all_tel["Speed"].to_numpy(dtype=float)
            thr_raw = all_tel["Throttle"].to_numpy(dtype=float)
            brk_raw = all_tel["Brake"].to_numpy(dtype=float)
            ger_raw = all_tel["nGear"].to_numpy(dtype=float)
            
            drv_session_times = drv_laps["Time"].dt.total_seconds().to_numpy()
            drv_lap_nums = drv_laps["LapNumber"].to_numpy()
            drv_compounds = drv_laps["Compound"].to_numpy()

            if len(t_raw) < 2:
                continue

            t_start = float(t_raw[0])
            t_end   = min(float(t_raw[-1]), MAX_DURATION_S)

            t_samples = np.arange(t_start, t_end, SAMPLE_INTERVAL)
            x_samples = np.interp(t_samples, t_raw, x_raw)
            y_samples = np.interp(t_samples, t_raw, y_raw)
            s_samples = np.interp(t_samples, t_raw, s_raw)
            thr_samples = np.interp(t_samples, t_raw, thr_raw)
            brk_samples = np.interp(t_samples, t_raw, brk_raw)
            ger_samples = np.round(np.interp(t_samples, t_raw, ger_raw))
            
            # For each t_sample, find which lap it belongs to (SessionTime is lap end time)
            lap_indices = np.searchsorted(drv_session_times, t_samples)
            lap_indices = np.clip(lap_indices, 0, len(drv_laps) - 1)
            
            comp_samples = drv_compounds[lap_indices]
            lap_samples = drv_lap_nums[lap_indices]

            for i in range(len(t_samples)):
                t_s, x_s, y_s, c_s, l_s = t_samples[i], x_samples[i], y_samples[i], comp_samples[i], lap_samples[i]
                t_key = round(t_s, 1)
                state = {
                    "code":  code,
                    "x":     round(float(x_s), 1),
                    "y":     round(float(y_s), 1),
                    "s":     int(s_samples[i]),
                    "t":     int(thr_samples[i]),
                    "b":     int(brk_samples[i]),
                    "g":     int(ger_samples[i]),
                    "tyre":  str(c_s)[0] if c_s and str(c_s) != "nan" else "?",
                    "lap":   int(l_s) if l_s and str(l_s) != "nan" else 0,
                }
                driver_frames.setdefault(t_key, []).append(state)
        except Exception as ex:
            print(f"Driver {drv_num} error: {ex}")

    all_times = sorted(driver_frames.keys())
    if not all_times:
        raise ValueError("No position data found for this session")

    t_min = all_times[0]
    t_max = all_times[-1]

    frames = [driver_frames[t] for t in all_times]

    total_laps = 0
    try:
        total_laps = int(sess.laps["LapNumber"].max())
    except Exception:
        pass

    result = {
        "year":        year,
        "race":        race,
        "session":     session_type,
        "track_x":     track_x,
        "track_y":     track_y,
        "total_laps":  total_laps,
        "t_min":       t_min,
        "t_max":       t_max,
        "frame_step":  SAMPLE_INTERVAL,
        "frames":      frames,
    }

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass

    return result

class handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        qs     = parse_qs(parsed.query)

        year_str     = qs.get("year",    ["2025"])[0]
        race         = qs.get("race",    ["Australian Grand Prix"])[0]
        session_type = qs.get("session", ["R"])[0].upper()

        try:
            year = int(year_str)
        except ValueError:
            self._send(400, json.dumps({"error": "Invalid year"}))
            return

        if session_type not in ("R", "S", "Q"):
            session_type = "R"

        try:
            data = build_replay_data(year, race, session_type)
            self._send(200, json.dumps(data))
        except ValueError as exc:
            self._send(404, json.dumps({"error": str(exc)}))
        except Exception as exc:
            self._send(500, json.dumps({"error": str(exc)}))

    def _send(self, code: int, body: str):
        encoded = body.encode()
        self.send_response(code)
        self.send_header("Content-Type",  "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt, *args):  # silence default access logs
        pass
