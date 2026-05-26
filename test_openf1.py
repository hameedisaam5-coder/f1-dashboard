import requests
try:
    print("Fetching sessions...")
    resp = requests.get('https://api.openf1.org/v1/sessions?year=2026', timeout=10)
    print("Status:", resp.status_code)
    data = resp.json()
    for s in data:
        print(s.get('session_name'), s.get('date_start'))
except Exception as e:
    print("Error:", e)
