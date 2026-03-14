"""
Vercel serverless function: GET /api/replay_data
Returns historical race replay frames as compact JSON.
Usage: ?year=2025&race=Bahrain+Grand+Prix&session=R
"""

import os
import json
import time
import pickle
import hashlib
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler

import fastf1
import pandas as pd
import numpy as np

CACHE_DIR = "/tmp/fastf1_cache"
REPLAY_CACHE_DIR = "/tmp/replay_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(REPLAY_CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def build_replay_data(year: int, race: str, session_type: str) -> dict:
    """
    Load FastF1 session and build a compact 1 Hz replay frame list.
    Samples car positions once per second across the whole session.
    """
    cache_key = hashlib.md5(f"{year}:{race}:{session_type}".encode()).hexdigest()
    cache_file = os.path.join(REPLAY_CACHE_DIR, f"{cache_key}.pkl")

    # Return from disk cache if available
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    sess = fastf1.get_session(year, race, session_type)
    sess.load(telemetry=True, weather=False, messages=False)

    # Build track layout from fastest lap
    track_x, track_y = [], []
    try:
        fastest = sess.laps.pick_fastest()
        tel = fastest.get_telemetry()
        track_x = tel["X"].tolist()
        track_y = tel["Y"].tolist()
    except Exception as e:
        print("Track layout error:", e)

    # Get driver list
    drivers = {}
    for drv_num in sess.drivers:
        try:
            drv_laps = sess.laps.pick_drivers(drv_num)
            if drv_laps.empty:
                continue
            code = drv_laps.iloc[0]["Driver"]
            drivers[drv_num] = code
        except Exception:
            pass

    # Find global session time range
    t_min = None
    t_max = None
    for drv_num, code in drivers.items():
        try:
            drv_laps = sess.laps.pick_drivers(drv_num)
            for _, lap in drv_laps.iterlaps():
                tel = lap.get_telemetry()
                if tel.empty:
                    continue
                t_arr = tel["SessionTime"].dt.total_seconds()
                t0, t1 = t_arr.min(), t_arr.max()
                if t_min is None or t0 < t_min:
                    t_min = t0
                if t_max is None or t1 > t_max:
                    t_max = t1
        except Exception:
            pass

    if t_min is None:
        raise ValueError("No telemetry data found for this session")

    # Build per-driver telemetry indexed by time
    drv_data = {}
    for drv_num, code in drivers.items():
        try:
            drv_laps = sess.laps.pick_drivers(drv_num)
            t_all, x_all, y_all, lap_all, tyre_all, speed_all = [], [], [], [], [], []
            for _, lap in drv_laps.iterlaps():
                tel = lap.get_telemetry()
                if tel.empty:
                    continue
                t_sec = tel["SessionTime"].dt.total_seconds().to_numpy()
                t_all.append(t_sec)
                x_all.append(tel["X"].to_numpy())
                y_all.append(tel["Y"].to_numpy())
                lap_num = float(lap["LapNumber"]) if pd.notna(lap["LapNumber"]) else 0.0
                lap_all.append(np.full(len(t_sec), lap_num))
                compound = str(lap["Compound"]) if pd.notna(lap.get("Compound", None)) else "UNKNOWN"
                tyre_all.append(np.full(len(t_sec), ord(compound[0]) if compound else 85))
                speed_all.append(tel["Speed"].to_numpy() if "Speed" in tel.columns else np.zeros(len(t_sec)))

            if not t_all:
                continue

            t_arr = np.concatenate(t_all)
            order = np.argsort(t_arr)
            t_arr = t_arr[order]
            x_arr = np.concatenate(x_all)[order]
            y_arr = np.concatenate(y_all)[order]
            lap_arr = np.concatenate(lap_all)[order]
            tyre_arr = np.concatenate(tyre_all)[order]
            speed_arr = np.concatenate(speed_all)[order]

            drv_data[code] = {
                "t": t_arr,
                "x": x_arr,
                "y": y_arr,
                "lap": lap_arr,
                "tyre": tyre_arr,
                "speed": speed_arr,
            }
        except Exception as e:
            print(f"Driver {code} error: {e}")

    # Build 1 Hz frames
    total_secs = int(t_max - t_min)
    total_laps = 0
    try:
        total_laps = int(sess.laps["LapNumber"].max())
    except Exception:
        pass

    frames = []
    # Limit to the first 7200 seconds (2 hours) for very long sessions
    frame_count = min(total_secs, 7200)

    for i in range(0, frame_count, 1):
        t = t_min + i
        drv_states = []
        for code, d in drv_data.items():
            idx = int(np.searchsorted(d["t"], t, side="right")) - 1
            if idx < 0 or idx >= len(d["t"]):
                continue
            # Only include if within 5 seconds of a real measurement
            if abs(d["t"][idx] - t) > 5:
                continue
            tyre_char = chr(int(d["tyre"][idx])) if d["tyre"][idx] >= 65 else "?"
            lap_num = int(d["lap"][idx])
            drv_states.append({
                "code": code,
                "x": round(float(d["x"][idx]), 1),
                "y": round(float(d["y"][idx]), 1),
                "lap": lap_num,
                "tyre": tyre_char,
                "speed": int(d["speed"][idx]) if d["speed"][idx] > 0 else 0,
            })
        frames.append(drv_states)

    result = {
        "year": year,
        "race": race,
        "session": session_type,
        "track_x": track_x,
        "track_y": track_y,
        "total_laps": total_laps,
        "t_min": t_min,
        "t_max": t_max,
        "frames": frames,
    }

    # Cache to disk
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass

    return result


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        year_str    = qs.get("year", ["2025"])[0]
        race        = qs.get("race", ["Bahrain Grand Prix"])[0]
        session_type = qs.get("session", ["R"])[0].upper()

        try:
            year = int(year_str)
        except ValueError:
            self._send(400, json.dumps({"error": "Invalid year"}))
            return

        if session_type not in ["R", "S", "Q"]:
            session_type = "R"

        try:
            data = build_replay_data(year, race, session_type)
            self._send(200, json.dumps(data))
        except ValueError as e:
            self._send(404, json.dumps({"error": str(e)}))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass
