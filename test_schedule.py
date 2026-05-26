import fastf1
import datetime
today = datetime.date.today()
year = today.year
print(year)

schedule = fastf1.get_event_schedule(year, include_testing=False)
print("Schedule:")
print(schedule[["EventName", "EventDate"]])
