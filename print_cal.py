import fastf1
import datetime

schedule = fastf1.get_event_schedule(2026, include_testing=False)
today = datetime.date.today()
with open("cal.txt", "w", encoding="utf-8") as f:
    f.write(f"Today is {today}\n")
    for _, row in schedule.iterrows():
        ed = row['EventDate']
        d = ed.date() if hasattr(ed, 'date') else ed
        delta = (d - today).days
        f.write(f"{row['EventName']} - Date: {d} - Delta: {delta}\n")
