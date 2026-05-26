import fastf1

fastf1.Cache.enable_cache("cache")
print("Getting session...")
try:
    sess = fastf1.get_session(2026, "Chinese Grand Prix", "R")
    sess.load()
    print("Event:", sess.event.EventName)
    print("Drivers:", sess.drivers)
    print("Laps:", len(sess.laps))
except Exception as e:
    print("Error:", e)
