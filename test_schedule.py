import fastf1
import datetime

today = datetime.date.today()
year  = today.year

print(f"Testing schedule detection for today: {today}")

try:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    # EventDate is the race Sunday; look for events ±7 days from today
    found = False
    for _, row in schedule.iterrows():
        event_date = row["EventDate"]
        if hasattr(event_date, "date"):
            event_date = event_date.date()
        delta = (event_date - today).days
        print(f"Event: {row['EventName']} | Date: {event_date} | Delta: {delta} days")
        if -3 <= delta <= 4:          # race week window
            print(f">>> CURRENT ACTIVE EVENT: {row['EventName']}")
            found = True
            break
            
    if not found:
        print("No active event found within -7 to +4 days window.")
        # Fallback test
        past = schedule[schedule["EventDate"].apply(
            lambda d: (d.date() if hasattr(d, "date") else d) < today
        )]
        if not past.empty:
            row = past.iloc[-1]
            print(f"Fallback event: {row['EventName']}")
            
except Exception as e:
    print("Schedule lookup failed:", e)
