import fastf1
import datetime
import io

schedule = fastf1.get_event_schedule(2026, include_testing=False)
today = datetime.date.today()

with io.open("cal.txt", "w", encoding="utf-8") as f:
    f.write(f"Today is: {today}\n")
    for _, row in schedule.head(5).iterrows():
        ed = row['EventDate']
        if hasattr(ed, 'date'): ed = ed.date()
        delta = (ed - today).days
        f.write(f"Event: {row['EventName']}\n")
        f.write(f"  EventDate: {ed} (Delta: {delta} days)\n")
        f.write(f"  EventFormat: {row['EventFormat']}\n")
        f.write("---\n")
