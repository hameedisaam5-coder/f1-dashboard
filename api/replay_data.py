"""
Vercel serverless function: GET /api/replay_data
Returns historical race replay as complete JSON with Telemetry, Weather, and Safety Car simulation.
"""

import os
import json
import pickle
import hashlib
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler

import fastf1
import numpy as np

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

CACHE_DIR   = "/tmp/fastf1_cache"
REPLAY_DIR  = "/tmp/replay_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(REPLAY_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def _compute_safety_car_positions(frames, track_statuses, session):
    if not frames or not track_statuses or not HAS_SCIPY:
        return

    try:
        fastest_lap = session.laps.pick_fastest()
        if fastest_lap is None:
            return
        tel = fastest_lap.get_telemetry()
        if tel is None or tel.empty:
            return
        
        ref_xs = tel["X"].to_numpy().astype(float)
        ref_ys = tel["Y"].to_numpy().astype(float)
        ref_dist = tel["Distance"].to_numpy().astype(float)
        
        if len(ref_xs) < 10: return
        
        t_old = np.linspace(0, 1, len(ref_xs))
        t_new = np.linspace(0, 1, 4000)
        ref_xs_dense = np.interp(t_new, t_old, ref_xs)
        ref_ys_dense = np.interp(t_new, t_old, ref_ys)
        
        ref_tree = cKDTree(np.column_stack((ref_xs_dense, ref_ys_dense)))
        
        diffs = np.sqrt(np.diff(ref_xs_dense)**2 + np.diff(ref_ys_dense)**2)
        ref_cumdist = np.concatenate(([0.0], np.cumsum(diffs)))
        ref_total = float(ref_cumdist[-1])
        
        dx = np.gradient(ref_xs_dense)
        dy = np.gradient(ref_ys_dense)
        norm = np.sqrt(dx**2 + dy**2)
        norm[norm == 0] = 1.0
        ref_nx = -dy / norm
        ref_ny = dx / norm
        
    except Exception:
        return

    sc_periods = []
    for status in track_statuses:
        if str(status.get("status", "")) == "4":
            sc_periods.append({
                "start_time": status.get("start_time", 0),
                "end_time": status.get("end_time"),
            })
    
    if not sc_periods: return

    DEPLOY_PIT_EXIT_DURATION = 4.0
    DEPLOY_CRUISE_SPEED = 55.0
    DEPLOY_TOTAL_MAX = 120.0
    SC_OFFSET_METERS = 150
    RETURN_ACCEL_DURATION = 5.0
    RETURN_ACCEL_SPEED = 400.0
    RETURN_PIT_ENTER_DURATION = 3.0
    RETURN_TOTAL = RETURN_ACCEL_DURATION + RETURN_PIT_ENTER_DURATION
    PIT_OFFSET_INWARD = 400

    def _pos_at_dist(dist_m):
        d = dist_m % ref_total
        idx = int(np.searchsorted(ref_cumdist, d))
        idx = min(idx, len(ref_xs_dense) - 1)
        return float(ref_xs_dense[idx]), float(ref_ys_dense[idx])
    
    def _idx_at_dist(dist_m):
        d = dist_m % ref_total
        idx = int(np.searchsorted(ref_cumdist, d))
        return min(idx, len(ref_xs_dense) - 1)
    
    def _dist_of_point(x, y):
        _, idx = ref_tree.query([x, y])
        return float(ref_cumdist[int(idx)])

    pit_exit_track_dist = ref_total * 0.05
    pit_exit_idx = _idx_at_dist(pit_exit_track_dist)
    pit_exit_track_x, pit_exit_track_y = _pos_at_dist(pit_exit_track_dist)
    pit_exit_pit_x = float(ref_xs_dense[pit_exit_idx] + ref_nx[pit_exit_idx] * PIT_OFFSET_INWARD)
    pit_exit_pit_y = float(ref_ys_dense[pit_exit_idx] + ref_ny[pit_exit_idx] * PIT_OFFSET_INWARD)
    
    pit_entry_track_dist = ref_total * 0.95
    pit_entry_idx = _idx_at_dist(pit_entry_track_dist)
    pit_entry_track_x, pit_entry_track_y = _pos_at_dist(pit_entry_track_dist)
    pit_entry_pit_x = float(ref_xs_dense[pit_entry_idx] + ref_nx[pit_entry_idx] * PIT_OFFSET_INWARD)
    pit_entry_pit_y = float(ref_ys_dense[pit_entry_idx] + ref_ny[pit_entry_idx] * PIT_OFFSET_INWARD)

    def get_leader_info(drv_list):
        if not drv_list: return None, None, None, None, None
        best = max(drv_list, key=lambda d: ((d.get("lap",1)-1)*ref_total + d.get("d",0)))
        return best["code"], best["x"], best["y"], _dist_of_point(best["x"], best["y"]), ((best.get("lap",1)-1)*ref_total + best.get("d",0))

    sc_state = {}
    
    for frame in frames: # frame has {"t": time, "drivers": [d1, d2...]}
        t = frame["t"]
        
        active_sc = None
        active_sc_idx = None
        for sci, sc in enumerate(sc_periods):
            sc_start = sc["start_time"]
            sc_end = sc.get("end_time")
            effective_end = (sc_end + RETURN_TOTAL) if sc_end else None
            if t >= sc_start and (effective_end is None or t < effective_end):
                active_sc = sc
                active_sc_idx = sci
                break
        
        if active_sc is None:
            frame["sc"] = None
            continue
            
        sc_start = active_sc["start_time"]
        sc_end = active_sc.get("end_time")
        elapsed = t - sc_start
        
        if active_sc_idx not in sc_state:
            sc_state[active_sc_idx] = {
                "track_dist": pit_exit_track_dist,
                "caught_up": False,
                "last_t": t,
                "return_start_dist": None,
                "prev_leader_dist": None,
            }
        
        state = sc_state[active_sc_idx]
        dt_frame = max(0.0, t - state["last_t"])
        state["last_t"] = t
        
        leader_code, leader_x, leader_y, leader_dist, leader_progress = get_leader_info(frame["drivers"])

        if elapsed < DEPLOY_PIT_EXIT_DURATION:
            phase = "deploying"
            progress = elapsed / DEPLOY_PIT_EXIT_DURATION
            alpha = progress
            smooth_t = 0.5 - 0.5 * np.cos(progress * np.pi)
            sc_x = pit_exit_pit_x + smooth_t * (pit_exit_track_x - pit_exit_pit_x)
            sc_y = pit_exit_pit_y + smooth_t * (pit_exit_track_y - pit_exit_pit_y)
            
        elif elapsed < DEPLOY_PIT_EXIT_DURATION + DEPLOY_TOTAL_MAX and not state["caught_up"]:
            phase = "deploying"
            alpha = 1.0
            
            if leader_code is not None:
                if state["prev_leader_dist"] is not None and dt_frame > 0:
                    leader_moved = leader_dist - state["prev_leader_dist"]
                    if leader_moved > ref_total / 2: leader_moved -= ref_total
                    elif leader_moved < -ref_total / 2: leader_moved += ref_total
                    leader_speed = abs(leader_moved) / dt_frame
                else:
                    leader_speed = 55.0
                state["prev_leader_dist"] = leader_dist
                sc_speed = max(20.0, min(leader_speed * 0.8, 60.0))
            else:
                sc_speed = DEPLOY_CRUISE_SPEED
            
            state["track_dist"] = (state["track_dist"] + sc_speed * dt_frame) % ref_total
            sc_x, sc_y = _pos_at_dist(state["track_dist"])
            
            if leader_code is not None:
                gap_ahead = (state["track_dist"] - leader_dist) % ref_total
                if gap_ahead <= SC_OFFSET_METERS + 50:
                    state["caught_up"] = True
                    
        elif sc_end is not None and t >= sc_end:
            return_elapsed = t - sc_end
            if state["return_start_dist"] is None:
                state["return_start_dist"] = state["track_dist"]
            
            if return_elapsed < RETURN_ACCEL_DURATION:
                phase = "returning"
                alpha = 1.0
                state["track_dist"] = (state["track_dist"] + RETURN_ACCEL_SPEED * dt_frame) % ref_total
                sc_x, sc_y = _pos_at_dist(state["track_dist"])
            else:
                phase = "returning"
                pit_enter_elapsed = return_elapsed - RETURN_ACCEL_DURATION
                progress = min(1.0, pit_enter_elapsed / RETURN_PIT_ENTER_DURATION)
                alpha = max(0.0, 1.0 - progress)
                track_x, track_y = _pos_at_dist(state["track_dist"])
                smooth_t = 0.5 - 0.5 * np.cos(progress * np.pi)
                sc_x = track_x + smooth_t * (pit_entry_pit_x - track_x)
                sc_y = track_y + smooth_t * (pit_entry_pit_y - track_y)
        else:
            phase = "on_track"
            alpha = 1.0
            state["caught_up"] = True
            
            if leader_code is not None:
                target_dist = (leader_dist + SC_OFFSET_METERS) % ref_total
                state["track_dist"] = target_dist
            else:
                state["track_dist"] = (state["track_dist"] + 100.0 * dt_frame) % ref_total
            sc_x, sc_y = _pos_at_dist(state["track_dist"])
        
        frame["sc"] = {"x": round(sc_x, 1), "y": round(sc_y, 1), "phase": phase, "alpha": round(alpha, 2)}

def build_replay_data(year: int, race: str, session_type: str) -> dict:
    cache_key  = hashlib.md5(f"{year}:{race}:{session_type}:v7".encode()).hexdigest()
    cache_file = os.path.join(REPLAY_DIR, f"{cache_key}.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Load session
    sess = fastf1.get_session(year, race, session_type)
    sess.load(telemetry=True, weather=True, messages=False)

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

    driver_data = {}
    global_t_min = None

    for drv_num in sess.drivers:
        try:
            drv_laps = sess.laps.pick_drivers(drv_num)
            if drv_laps.empty: continue
            code = drv_laps.iloc[0]["Driver"]

            all_tel = drv_laps.get_telemetry()
            if all_tel.empty or "X" not in all_tel.columns: continue

            t_raw = all_tel["SessionTime"].dt.total_seconds().to_numpy()
            x_raw = all_tel["X"].to_numpy(dtype=float)
            y_raw = all_tel["Y"].to_numpy(dtype=float)
            s_raw = all_tel["Speed"].to_numpy(dtype=float)
            thr_raw = all_tel["Throttle"].to_numpy(dtype=float)
            
            if all_tel["Brake"].dtype == bool:
                brk_raw = all_tel["Brake"].to_numpy(dtype=int) * 100
            else:
                brk_raw = all_tel["Brake"].to_numpy(dtype=float)
                
            ger_raw = all_tel["nGear"].to_numpy(dtype=float)
            
            if "Distance" in all_tel.columns:
                dst_raw = all_tel["Distance"].to_numpy(dtype=float)
            else:
                dst_raw = np.zeros(len(t_raw))
            
            drv_session_times = drv_laps["Time"].dt.total_seconds().to_numpy()
            drv_lap_nums = drv_laps["LapNumber"].to_numpy()
            drv_compounds = drv_laps["Compound"].to_numpy()

            if len(t_raw) < 2: continue

            driver_data[code] = {
                "t": t_raw, "x": x_raw, "y": y_raw, "s": s_raw, 
                "thr": thr_raw, "brk": brk_raw, "ger": ger_raw, "dst": dst_raw,
                "laps": (drv_session_times, drv_lap_nums, drv_compounds)
            }
            if global_t_min is None or t_raw[0] < global_t_min:
                global_t_min = t_raw[0]
        except Exception as ex:
            pass

    if not driver_data:
        raise ValueError("No positive data found for drivers")

    # Time bounds relative to global start
    all_t_raws = np.concatenate([d["t"] for d in driver_data.values()])
    t_start = 0.0
    t_end = min(np.max(all_t_raws) - global_t_min, MAX_DURATION_S)
    timeline = np.arange(t_start, t_end, SAMPLE_INTERVAL)
    
    # Track Status Parsing
    track_statuses = []
    if hasattr(sess, 'track_status') and not sess.track_status.empty:
        for status in sess.track_status.to_dict("records"):
            start_time = status["Time"].total_seconds() - global_t_min
            if track_statuses:
                track_statuses[-1]["end_time"] = start_time
            track_statuses.append({
                "status": status["Status"],
                "start_time": start_time,
                "end_time": None
            })

    # Weather Parsing
    weather_times = []
    weather_dict = {"air_temp":[], "track_temp":[], "humidity":[], "wind_speed":[], "wind_direction":[], "rainfall":[]}
    if hasattr(sess, 'weather_data') and not sess.weather_data.empty:
        weather_df = sess.weather_data
        weather_times = weather_df["Time"].dt.total_seconds().to_numpy() - global_t_min
        
        def _get_w(col): return weather_df[col].to_numpy() if col in weather_df else np.nan*np.ones_like(weather_times)
        
        weather_dict["air_temp"] = _get_w("AirTemp")
        weather_dict["track_temp"] = _get_w("TrackTemp")
        weather_dict["humidity"] = _get_w("Humidity")
        weather_dict["wind_speed"] = _get_w("WindSpeed")
        weather_dict["wind_direction"] = _get_w("WindDirection")
        weather_dict["rainfall"] = _get_w("Rainfall")

    # Build pure frames structure combining all
    raw_frames = [{"t": float(t), "drivers": []} for t in timeline]

    for code, d in driver_data.items():
        t_shifted = d["t"] - global_t_min
        x_s = np.interp(timeline, t_shifted, d["x"])
        y_s = np.interp(timeline, t_shifted, d["y"])
        s_s = np.interp(timeline, t_shifted, d["s"])
        thr_s = np.interp(timeline, t_shifted, d["thr"])
        brk_s = np.interp(timeline, t_shifted, d["brk"])
        ger_s = np.round(np.interp(timeline, t_shifted, d["ger"]))
        dst_s = np.interp(timeline, t_shifted, d["dst"])
        
        drv_session_times, drv_lap_nums, drv_compounds = d["laps"]
        drv_session_times_shifted = drv_session_times - global_t_min
        
        lap_indices = np.searchsorted(drv_session_times_shifted, timeline)
        lap_indices = np.clip(lap_indices, 0, len(drv_lap_nums) - 1)
        comp_samples = drv_compounds[lap_indices]
        lap_samples = drv_lap_nums[lap_indices]

        for i, t_val in enumerate(timeline):
            raw_frames[i]["drivers"].append({
                "code": code,
                "x": round(float(x_s[i]), 1),
                "y": round(float(y_s[i]), 1),
                "s": int(s_s[i]),
                "t": int(thr_s[i]),
                "b": int(brk_s[i]),
                "g": int(ger_s[i]),
                "d": float(dst_s[i]),
                "tyre": str(comp_samples[i])[0] if comp_samples[i] and str(comp_samples[i]) != "nan" else "?",
                "lap": int(lap_samples[i]) if lap_samples[i] and str(lap_samples[i]) != "nan" else 0,
            })

    # Weather injection
    for i, t_val in enumerate(timeline):
        if len(weather_times) > 0:
            idx = int(np.searchsorted(weather_times, t_val))
            idx = min(idx, len(weather_times)-1)
            raw_frames[i]["weather"] = {
                "air_temp": float(weather_dict["air_temp"][idx]),
                "track_temp": float(weather_dict["track_temp"][idx]),
                "humidity": float(weather_dict["humidity"][idx]),
                "wind_speed": float(weather_dict["wind_speed"][idx]),
                "wind_direction": float(weather_dict["wind_direction"][idx]),
                "rainfall": float(weather_dict["rainfall"][idx]),
            }

    # Safety car injection (modifies raw_frames in place)
    _compute_safety_car_positions(raw_frames, track_statuses, sess)

    total_laps = 0
    try: total_laps = int(sess.laps["LapNumber"].max())
    except Exception: pass

    result = {
        "year":        year,
        "race":        race,
        "session":     session_type,
        "track_x":     track_x,
        "track_y":     track_y,
        "total_laps":  total_laps,
        "t_min":       0,
        "t_max":       t_end,
        "frame_step":  SAMPLE_INTERVAL,
        "frames":      raw_frames,
    }

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    except Exception: pass

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
            # Remove NaN values by dumping with a custom function or just relying on json handling
            # Python json doesn't support NaNs officially if client rejects it, let's just dump
            clean_data_str = json.dumps(data, allow_nan=False).replace("NaN", "null")
            self._send(200, clean_data_str, direct=True)
        except ValueError as exc:
            self._send(404, json.dumps({"error": str(exc)}))
        except Exception as exc:
            self._send(500, json.dumps({"error": str(exc)}))

    def _send(self, code: int, body: str, direct=False):
        try:
            encoded = body if direct and isinstance(body, bytes) else body.encode()
        except:
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
