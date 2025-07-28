import requests
import csv
from datetime import datetime
import calendar

# === EDIT THESE ===
year = 2025   # Change this
month = 7     # Change this (1 = Jan, 12 = Dec)
# ==================

# Calculate start and end timestamps in milliseconds
start_date = datetime(year, month, 1)
last_day = calendar.monthrange(year, month)[1]
end_date = datetime(year, month, last_day, 23, 59, 59)

params = {
    "game-names": "CASH 3",
    "date-from": int(start_date.timestamp() * 1000),
    "date-to": int(end_date.timestamp() * 1000),
    "status": "CLOSED",
    "order": "desc",
    "page": 0,
    "size": 20
}

# API endpoint
base_url = "https://www.galottery.com/api/v2/draw-games/draws/page"
output_file = f"cash3_results_{year}_{month}.csv"

all_results = []

while True:
    response = requests.get(base_url, params=params)
    data = response.json()

    if "draws" not in data or not data["draws"]:
        break

    for draw in data["draws"]:
        # Extract date
        draw_time = draw.get("drawTime")
        date = datetime.fromtimestamp(draw_time / 1000).strftime('%m/%d/%Y')

        # Draw name (MIDDAY/EVENING/NIGHT)
        draw_name = draw.get("name", "")

        # Winning numbers (zero-padded to always be 3 digits)
        raw_digits = draw.get("results", [{}])[0].get("primary", [])
        if raw_digits:
            numbers = f"{int(''.join(raw_digits)):03d}"
        else:
            numbers = "000"  # Fallback if no digits found

        all_results.append([date, draw_name, numbers])

    if "nextPageUrl" not in data or not data["nextPageUrl"]:
        break

    params["page"] += 1

draw_order = {"NIGHT": 3, "EVENING": 2, "MIDDAY": 1}

all_results.sort(
    key=lambda x: (
        datetime.strptime(x[0], '%m/%d/%Y'),
        -draw_order.get(x[1], 4)
    ),
    reverse=True
)

with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Date", "Draw", "Winning_Numbers"])
    writer.writerows(all_results)

print(f"✅ Data saved to {output_file}")
