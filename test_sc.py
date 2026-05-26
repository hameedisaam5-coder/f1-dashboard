import sys
import json
from api.replay_data import build_replay_data

try:
    print("Building replay data for 2025 AUS Race...")
    data = build_replay_data(2025, "Australian Grand Prix", "R")
    
    frames = data["frames"]
    print(f"Total Frames: {len(frames)}")
    if not frames:
        print("No frames returned!")
        sys.exit(1)
        
    print("Testing JSON serialization of frame 50...")
    s = json.dumps(frames[50], allow_nan=False).replace("NaN", "null")
    print(len(s), "bytes")
    
    print("\nSample Frame SC:", frames[100].get("sc"))
    print("Sample Frame Weather:", frames[50].get("weather"))
    
    print("SUCCESS")
except Exception as e:
    print("FAILED", type(e), str(e))
    import traceback
    traceback.print_exc()
