import fastf1
import pandas as pd
import json
import time
import sys

fastf1.Cache.enable_cache("cache")

YEAR = 2026
RACE = "Australia"
SESSION = sys.argv[1].upper() if len(sys.argv) > 1 else "R"

print("Loading session...")
session = fastf1.get_session(YEAR, RACE, SESSION)
session.load()
print("Session loaded")

try:
    fastest  = session.laps.pick_fastest()
    tel      = fastest.get_telemetry()
    track_x  = tel["X"].tolist()
    track_y  = tel["Y"].tolist()
except Exception as e:
    print("Track layout error:", e)
    track_x, track_y = [], []


def is_retired(session, driver_number):
    try:
        if hasattr(session, "results") and not session.results.empty:
            dr = session.results[
                session.results["DriverNumber"].astype(str) == str(driver_number)
            ]
            if not dr.empty:
                st = str(dr.iloc[0].get("Status", "") or "")
                if st and st not in ["Finished", ""] and not st.startswith("+"):
                    return True, st
    except Exception as e:
        print("Retirement check error:", e)
    return False, None


def linear_slope(values):
    """Simple linear regression slope — no numpy needed."""
    n = len(values)
    if n < 2:
        return None
    sx  = n * (n - 1) / 2
    sy  = sum(values)
    sxy = sum(i * v for i, v in enumerate(values))
    sx2 = n * (n - 1) * (2 * n - 1) / 6
    denom = n * sx2 - sx * sx
    return (n * sxy - sx * sy) / denom if denom else None


def _secs(td):
    """Convert a pandas Timedelta to float seconds, or return None."""
    try:
        return round(float(td.total_seconds()), 3) if pd.notna(td) else None
    except:
        return None


while True:
    # Refresh live session data every cycle so we pick up new laps/telemetry
    try:
        session.load(livedata=None)   # safe no-op on finished sessions
    except Exception as e:
        print("Session reload warning:", e)

    drivers_data = []

    try:
        track_status       = session.track_status.iloc[-1]["Status"]
        safety_car         = track_status in ["4", "5"]
        virtual_safety_car = track_status in ["6", "7"]
        red_flag           = track_status in ["2"]
    except:
        safety_car = virtual_safety_car = red_flag = False

    for driver in session.drivers:
        try:
            laps            = session.laps.pick_drivers(driver)
            retired, reason = is_retired(session, driver)

            if laps.empty:
                if retired:
                    driver_code = "-"
                    try:
                        dr = session.results[
                            session.results["DriverNumber"].astype(str) == str(driver)
                        ]
                        if not dr.empty:
                            driver_code = str(dr.iloc[0].get("Abbreviation", driver))
                    except:
                        pass
                    drivers_data.append({
                        "position": None, "driver_code": driver_code,
                        "driver_number": driver, "lap_number": None,
                        "last_lap": None, "compound": None, "tyre_age": None,
                        "stint": 1, "lap_history": [], "stint_history": [],
                        "tyre_deg": None, "avg_pace_3": None,
                        "x": None, "y": None, "retired": True,
                        "retire_reason": reason, "dnf_lap": None,
                        "gap_to_leader": "-", "gap_to_ahead": "-",
                    })
                continue

            last_lap   = laps.iloc[-1]
            driver_code = last_lap["Driver"]
            position   = int(last_lap["Position"])   if not pd.isna(last_lap["Position"])  else None
            lap_number = int(last_lap["LapNumber"])  if not pd.isna(last_lap["LapNumber"]) else None
            compound   = last_lap.get("Compound", None)
            stint      = int(last_lap["Stint"])      if not pd.isna(last_lap["Stint"])      else 1
            tyre_age   = int(last_lap["TyreLife"])   if not pd.isna(last_lap["TyreLife"])   else None

            lap_time_str = None
            if pd.notna(last_lap["LapTime"]):
                total        = last_lap["LapTime"].total_seconds()
                lap_time_str = f"{int(total//60)}:{total%60:06.3f}"

            lap_history = [float(l.total_seconds()) for l in laps["LapTime"].dropna().tail(6)]

            # ── Stint history ──────────────────────────────────────────────
            stint_history = []
            try:
                for snum in sorted(laps["Stint"].dropna().unique()):
                    sl = laps[laps["Stint"] == snum].dropna(subset=["LapNumber"])
                    if sl.empty:
                        continue
                    sc = str(sl.iloc[0].get("Compound") or "UNKNOWN")
                    ss = int(sl["LapNumber"].min())
                    se = int(sl["LapNumber"].max())
                    stint_history.append({
                        "stint": int(snum), "compound": sc,
                        "start_lap": ss, "end_lap": se,
                        "laps_on_set": se - ss + 1,
                    })
            except:
                pass

            # ── Tyre degradation (s/lap) ───────────────────────────────────
            tyre_deg = None
            try:
                times = [float(l.total_seconds()) for l in laps["LapTime"].dropna().tail(8)]
                if len(times) >= 4:
                    slope = linear_slope(times)
                    if slope is not None:
                        tyre_deg = round(slope, 3)
            except:
                pass

            # ── Average pace last 3 laps ───────────────────────────────────
            avg_pace_3 = None
            try:
                p3 = [float(l.total_seconds()) for l in laps["LapTime"].dropna().tail(3)]
                if p3:
                    avg_pace_3 = round(sum(p3) / len(p3), 3)
            except:
                pass

            # ── Sector times (session best + last lap) ─────────────────────
            s1_best = s2_best = s3_best = None
            s1_last = s2_last = s3_last = None
            try:
                for scol in ["Sector1Time", "Sector2Time", "Sector3Time"]:
                    if scol not in laps.columns:
                        continue
                    valid = laps[scol].dropna()
                    lv    = last_lap.get(scol)
                    if scol == "Sector1Time":
                        if not valid.empty: s1_best = _secs(valid.min())
                        if lv is not None and pd.notna(lv): s1_last = _secs(lv)
                    elif scol == "Sector2Time":
                        if not valid.empty: s2_best = _secs(valid.min())
                        if lv is not None and pd.notna(lv): s2_last = _secs(lv)
                    elif scol == "Sector3Time":
                        if not valid.empty: s3_best = _secs(valid.min())
                        if lv is not None and pd.notna(lv): s3_last = _secs(lv)
            except:
                pass

            # ── Track position ─────────────────────────────────────────────
            x = y = None
            if not retired:
                try:
                    pd_data = session.pos_data[driver].dropna()
                    if not pd_data.empty:
                        latest = pd_data.iloc[-1]
                        x, y   = float(latest["X"]), float(latest["Y"])
                except:
                    pass

            drivers_data.append({
                "position": position, "driver_code": driver_code,
                "driver_number": driver, "lap_number": lap_number,
                "last_lap": lap_time_str, "compound": compound,
                "tyre_age": tyre_age, "stint": stint,
                "lap_history": lap_history, "stint_history": stint_history,
                "tyre_deg": tyre_deg, "avg_pace_3": avg_pace_3,
                "s1_best": s1_best, "s2_best": s2_best, "s3_best": s3_best,
                "s1_last": s1_last, "s2_last": s2_last, "s3_last": s3_last,
                "x": x, "y": y, "retired": retired,
                "retire_reason": reason,
                "dnf_lap": lap_number if retired else None,
                "gap_to_leader": "-", "gap_to_ahead": "-",
            })

        except Exception as e:
            print("Driver error:", driver, e)

    classified = sorted(
        [d for d in drivers_data if not d.get("retired") and d["position"] is not None],
        key=lambda d: d["position"]
    )
    dnf_drivers = [d for d in drivers_data if d.get("retired")]

    # ── Gap calculation: use cumulative race time for each driver ────────────
    # Build a map of {driver_code: total_race_time_seconds} from all laps
    gap_times = {}
    for d in classified:
        try:
            drv_laps = session.laps.pick_drivers(d["driver_number"])
            total = drv_laps["LapTime"].dropna().apply(lambda t: t.total_seconds()).sum()
            if total > 0:
                gap_times[d["driver_code"]] = total
        except:
            pass

    leader_total = None
    prev_total   = None
    for d in classified:
        code  = d["driver_code"]
        total = gap_times.get(code)
        if total is None:
            d["gap_to_leader"] = d["gap_to_ahead"] = "-"
            continue
        if leader_total is None:
            leader_total = prev_total = total
            d["gap_to_leader"] = d["gap_to_ahead"] = "-"
        else:
            d["gap_to_leader"] = round(total - leader_total, 3)
            d["gap_to_ahead"]  = round(total - prev_total, 3)
            prev_total = total

    for d in dnf_drivers:
        d["gap_to_leader"] = d["gap_to_ahead"] = "-"

    output = {
        "session":            str(session.event.EventName),
        "timestamp":          time.time(),
        "safety_car":         safety_car,
        "virtual_safety_car": virtual_safety_car,
        "red_flag":           red_flag,
        "track_x":            track_x,
        "track_y":            track_y,
        "drivers":            classified + dnf_drivers,
    }

    with open("race_data.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Data updated")
    time.sleep(5)