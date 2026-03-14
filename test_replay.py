from api.replay_data import build_replay_data
import time
t0 = time.time()
print("Starting bulk replay build...")
try:
    d = build_replay_data(2025, 'Australian Grand Prix', 'R')
    print(f"Success! Done in {time.time()-t0:.2f}s, frames: {len(d['frames'])}")
except Exception as e:
    import traceback
    traceback.print_exc()
