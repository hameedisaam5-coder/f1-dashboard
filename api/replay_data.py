"""
Vercel serverless function: GET /api/replay_data
Returns historical race replay as complete JSON with Telemetry, Weather, and Safety Car simulation.
"""

import os
import json
import pickle
import hashlib
import math
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler

import fastf1
import numpy as np
import pandas as pd

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

def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value

def _median(values):
    vals = sorted(v for v in values if v is not None and not math.isnan(float(v)))
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2)

def _stdev(values):
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))

def _build_race_analysis(driver_laps, total_laps):
    drivers = []
    for code, laps in driver_laps.items():
        timed = [lap for lap in laps if lap.get("time")]
        if not timed:
            continue

        times = [float(lap["time"]) for lap in timed]
        filtered_times = []
        med = _median(times)
        if med:
            filtered_times = [t for t in times if t <= med * 1.12]
        filtered_times = filtered_times or times

        stint_map = {}
        compound_laps = {}
        for lap in timed:
            stint = int(lap.get("stint") or 0)
            tyre = str(lap.get("tyre") or "UNKNOWN").upper()
            stint_map.setdefault(stint, {"stint": stint, "tyre": tyre, "laps": 0})
            stint_map[stint]["laps"] += 1
            compound_laps[tyre] = compound_laps.get(tyre, 0) + 1

        first = timed[0]
        last = timed[-1]
        recent = timed[-5:]
        prev = timed[-10:-5]
        recent_avg = sum(float(l["time"]) for l in recent) / len(recent) if recent else None
        prev_avg = sum(float(l["time"]) for l in prev) / len(prev) if prev else None
        trend = recent_avg - prev_avg if recent_avg is not None and prev_avg is not None else None
        final_lap = int(last.get("lap") or 0)
        final_time = float(last.get("t") or 0)

        drivers.append({
            "code": code,
            "completed_laps": final_lap,
            "classified": bool(total_laps and final_lap >= max(1, total_laps - 1)),
            "best_lap": min(times),
            "avg_pace": sum(filtered_times) / len(filtered_times),
            "median_pace": med,
            "recent_pace": recent_avg,
            "pace_trend": trend,
            "consistency": _stdev(filtered_times),
            "pits": max(0, len([s for s in stint_map if s]) - 1),
            "stints": sorted(stint_map.values(), key=lambda s: s["stint"]),
            "compound_laps": compound_laps,
            "first_lap_time": float(first.get("time") or 0),
            "final_time": final_time,
        })

    if not drivers:
        return {"drivers": [], "classification": [], "predictions": []}

    classification = sorted(
        drivers,
        key=lambda d: (-d["completed_laps"], d["final_time"])
    )
    for pos, item in enumerate(classification, 1):
        item["finish_position"] = pos

    pace_sorted = sorted(drivers, key=lambda d: d["avg_pace"])
    best_pace = pace_sorted[0]["avg_pace"]
    worst_pace = pace_sorted[-1]["avg_pace"]
    spread = max(0.001, worst_pace - best_pace)

    prediction_rows = []
    for item in drivers:
        pace_score = 1 - ((item["avg_pace"] - best_pace) / spread)
        consistency = item["consistency"] if item["consistency"] is not None else 3.5
        consistency_score = max(0, min(1, 1 - consistency / 4.5))
        reliability_score = 1 if item["classified"] else 0.35
        trend_score = 0.55
        if item["pace_trend"] is not None:
            trend_score = max(0, min(1, 0.55 - item["pace_trend"] / 3.0))
        finish_bonus = max(0, 1 - ((item.get("finish_position", len(drivers)) - 1) / max(1, len(drivers) - 1)))
        raw = (
            pace_score * 0.42
            + consistency_score * 0.20
            + reliability_score * 0.18
            + trend_score * 0.12
            + finish_bonus * 0.08
        )
        prediction_rows.append({
            "code": item["code"],
            "score": raw,
            "reason": _prediction_reason(item, pace_score, consistency_score, trend_score)
        })

    total_score = sum(max(0.001, r["score"]) for r in prediction_rows)
    predictions = sorted([
        {
            "code": r["code"],
            "win_chance": round(max(0.001, r["score"]) / total_score * 100, 1),
            "reason": r["reason"],
        }
        for r in prediction_rows
    ], key=lambda r: r["win_chance"], reverse=True)

    return {
        "drivers": sorted(drivers, key=lambda d: d.get("finish_position", 99)),
        "classification": [{"position": d["finish_position"], "code": d["code"], "laps": d["completed_laps"]} for d in classification],
        "predictions": predictions[:8],
    }

def _prediction_reason(item, pace_score, consistency_score, trend_score):
    parts = []
    if pace_score > 0.75:
        parts.append("front-running pace")
    elif pace_score < 0.35:
        parts.append("pace deficit")
    if consistency_score > 0.70:
        parts.append("stable lap times")
    elif consistency_score < 0.35:
        parts.append("variable lap times")
    if trend_score > 0.65:
        parts.append("improving late pace")
    elif trend_score < 0.45:
        parts.append("fading late pace")
    if not item["classified"]:
        parts.append("reliability risk")
    return ", ".join(parts[:3]) or "balanced replay form"

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
    cache_key  = hashlib.md5(f"{year}:{race}:{session_type}:v9".encode()).hexdigest()
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
                
            drs_raw = all_tel["DRS"].to_numpy(dtype=float) if "DRS" in all_tel.columns else np.zeros(len(t_raw))
            
            drv_session_times = drv_laps["Time"].dt.total_seconds().to_numpy()
            drv_lap_nums = drv_laps["LapNumber"].to_numpy()
            drv_compounds = drv_laps["Compound"].to_numpy()

            # Build full lap history for strategy 
            lap_records = []
            for _, rl in drv_laps.iterrows():
                if pd.isna(rl["LapTime"]):
                    continue
                lap_records.append({
                    "lap": int(rl["LapNumber"]) if pd.notna(rl["LapNumber"]) else 0,
                    "t": float(rl["Time"].total_seconds()) if pd.notna(rl["Time"]) else 0,
                    "time": float(rl["LapTime"].total_seconds()),
                    "tyre": str(rl["Compound"]),
                    "age": int(rl["TyreLife"]) if pd.notna(rl["TyreLife"]) else 0,
                    "stint": int(rl["Stint"]) if pd.notna(rl["Stint"]) else 0,
                    "s1": float(rl["Sector1Time"].total_seconds()) if pd.notna(rl["Sector1Time"]) else None,
                    "s2": float(rl["Sector2Time"].total_seconds()) if pd.notna(rl["Sector2Time"]) else None,
                    "s3": float(rl["Sector3Time"].total_seconds()) if pd.notna(rl["Sector3Time"]) else None,
                })

            if len(t_raw) < 2: continue

            driver_data[code] = {
                "t": t_raw, "x": x_raw, "y": y_raw, "s": s_raw, 
                "thr": thr_raw, "brk": brk_raw, "ger": ger_raw, "dst": dst_raw, "drs": drs_raw,
                "laps": (drv_session_times, drv_lap_nums, drv_compounds),
                "lap_records": lap_records
            }
            if global_t_min is None or t_raw[0] < global_t_min:
                global_t_min = t_raw[0]
        except Exception as ex:
            pass

    if not driver_data:
        raise ValueError("No telemetry was available for this replay. Try another year, race, or session.")

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
        drs_s = np.round(np.interp(timeline, t_shifted, d["drs"]))
        
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
                "drs": int(drs_s[i]),
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

    driver_laps = {}
    for code, d in driver_data.items():
        driver_laps[code] = []
        for lap in d["lap_records"]:
            lap_copy = dict(lap)
            lap_copy["t"] = max(0.0, float(lap_copy.get("t", 0)) - global_t_min)
            driver_laps[code].append(lap_copy)

    race_analysis = _build_race_analysis(driver_laps, total_laps)

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
        "driver_laps": driver_laps,
        "race_analysis": race_analysis
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
            data = _json_safe(build_replay_data(year, race, session_type))
            clean_data_str = json.dumps(data, allow_nan=False)
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
