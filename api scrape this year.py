import requests
import csv
from datetime import datetime
import calendar
import time

# Function to get timestamp in milliseconds
def get_timestamp(year, month, day):
    dt = datetime(year, month, day, 0, 0)
    return int(time.mktime(dt.timetuple()) * 1000)

# Set the year for scraping
YEAR = 2025
GAME = "CASH 3"

base_url = "https://www.galottery.com/api/v2/draw-games/draws/page"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

all_results = []

# Loop through months
for month in range(1, 13):
    start_ts = get_timestamp(YEAR, month, 1)
    last_day = calendar.monthrange(YEAR, month)[1]
    end_ts = get_timestamp(YEAR, month, last_day)

    page = 0
    while True:
        params = {
            "game-names": GAME,
            "date-from": start_ts,
            "date-to": end_ts,
            "status": "CLOSED",
            "order": "desc",
            "page": page,
            "size": 20
        }

        r = requests.get(base_url, params=params, headers=headers)
        data = r.json()

        if "draws" not in data or not data["draws"]:
            break  # No more data for this month

        for draw in data["draws"]:
            draw_date = datetime.fromtimestamp(draw["drawTime"] / 1000).strftime("%Y-%m-%d")
            draw_time = draw["name"]
            numbers = "".join(draw["results"][0]["primary"]).zfill(3)  # Keep leading zeros
            all_results.append([draw_date, draw_time, numbers])

        page += 1

    print(f"✅ Fetched {month}/{YEAR}")

# Sort by date (latest first), then Night > Evening > Midday
draw_order = {"NIGHT": 3, "EVENING": 2, "MIDDAY": 1}
all_results.sort(key=lambda x: (x[0], draw_order.get(x[1].upper(), 99)), reverse=True)

# Save to CSV
filename = f"cash3_{YEAR}_sorted.csv"
with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Date", "Draw", "Winning_Numbers"])
    for row in all_results:
        writer.writerow([row[0], row[1], f"'{row[2]}"])  # Keep as text for Excel

print(f"✅ Completed. Total records: {len(all_results)}. Saved as {filename}")
