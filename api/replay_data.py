"""
Vercel serverless function: GET /api/replay_data
Returns historical race replay as compact per-lap position JSON.

Uses FastF1 laps+position data only (no heavy per-Hz telemetry),
sampled at 4Hz (every 0.25 s) to stay well within the 60-second timeout.
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
    """
    Build a compact replay dataset using only car position data.
    FastF1 provides position telemetry via laps; we sample every 2 seconds
    to keep the payload small and the processing time well under 60 s.
    """
    cache_key  = hashlib.md5(f"{year}:{race}:{session_type}:v4".encode()).hexdigest()
    cache_file = os.path.join(REPLAY_DIR, f"{cache_key}.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Load session – only laps + telemetry (no weather / messages)
    sess = fastf1.get_session(year, race, session_type)
    sess.load(telemetry=True, weather=False, messages=False)

    # ── Track layout from fastest lap ───────────────────────────────────────
    track_x, track_y = [], []
    try:
        fl  = sess.laps.pick_fastest()
        tel = fl.get_telemetry()
        # Downsample track outline to ≤2000 points
        step = max(1, len(tel) // 2000)
        track_x = tel["X"].iloc[::step].tolist()
        track_y = tel["Y"].iloc[::step].tolist()
    except Exception as ex:
        print("Track layout:", ex)

    # ── Per-driver position sampling ─────────────────────────────────────────
    # Strategy: iterate each driver's laps, get telemetry at ~2 Hz to avoid
    # loading millions of raw floats. Use numpy interp on the time axis.
    SAMPLE_INTERVAL = 2.0          # seconds between frames
    MAX_DURATION_S  = 7200         # 2 hours max

    driver_frames: dict[float, list] = {}   # t_sec -> list of driver states

    for drv_num in sess.drivers:
        try:
            drv_laps = sess.laps.pick_drivers(drv_num)
            if drv_laps.empty:
                continue
            code = drv_laps.iloc[0]["Driver"]

            for _, lap in drv_laps.iterlaps():
                try:
                    tel = lap.get_telemetry()
                except Exception:
                    continue
                if tel.empty or "X" not in tel.columns:
                    continue

                t_raw = tel["SessionTime"].dt.total_seconds().to_numpy()
                x_raw = tel["X"].to_numpy(dtype=float)
                y_raw = tel["Y"].to_numpy(dtype=float)

                if len(t_raw) < 2:
                    continue

                compound   = str(lap.get("Compound", "?") or "?")[0]
                lap_number = int(lap.get("LapNumber", 0) or 0)

                # Sample every SAMPLE_INTERVAL seconds within this lap
                t_start = float(t_raw[0])
                t_end   = min(float(t_raw[-1]), MAX_DURATION_S)

                t_samples = np.arange(t_start, t_end, SAMPLE_INTERVAL)
                x_samples = np.interp(t_samples, t_raw, x_raw)
                y_samples = np.interp(t_samples, t_raw, y_raw)

                for t_s, x_s, y_s in zip(t_samples, x_samples, y_samples):
                    t_key = round(t_s, 1)
                    state = {
                        "code":  code,
                        "x":     round(float(x_s), 1),
                        "y":     round(float(y_s), 1),
                        "tyre":  compound,
                        "lap":   lap_number,
                    }
                    driver_frames.setdefault(t_key, []).append(state)

        except Exception as ex:
            print(f"Driver {drv_num} error: {ex}")

    # ── Build ordered frame list ─────────────────────────────────────────────
    all_times = sorted(driver_frames.keys())
    if not all_times:
        raise ValueError("No position data found for this session")

    t_min = all_times[0]
    t_max = all_times[-1]

    # Sample at SAMPLE_INTERVAL cadence (already done above, just order them)
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
