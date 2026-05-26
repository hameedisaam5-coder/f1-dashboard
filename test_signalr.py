import fastf1
from fastf1.livetiming.client import SignalRClient
from fastf1.livetiming.data import LiveTimingData
import asyncio
import threading
import time

def run_client():
    asyncio.run(SignalRClient('live_test.txt', debug=False).start())

t = threading.Thread(target=run_client, daemon=True)
t.start()
print("Client started, waiting 10s...")
time.sleep(10)

try:
    print("Loading livedata...")
    livedata = LiveTimingData('live_test.txt')
    sess = fastf1.get_session(2026, "Chinese Grand Prix", "R")
    sess.load(livedata=livedata)
    print("Drivers with LiveData:", sess.drivers)
except Exception as e:
    print("Error:", e)
