import requests
import csv
from datetime import datetime

# API URL and params for July
url = "https://www.galottery.com/api/v2/draw-games/draws/page"
params = {
    "game-names": "CASH 3",
    "date-from": "1751342400000",
    "date-to": "1753502399999",
    "status": "CLOSED",
    "order": "desc",
    "page": 0,
    "size": 20
}

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

all_results = []

while True:
    r = requests.get(url, params=params, headers=headers)
    data = r.json()

    if "draws" not in data or not data["draws"]:
        break

    for draw in data["draws"]:
        draw_date = datetime.fromtimestamp(draw["drawTime"]/1000).strftime("%Y-%m-%d")
        draw_time = draw["name"]
        
        # FIX: join numbers and keep them as text with leading zeros
        numbers = "".join(draw["results"][0]["primary"])
        numbers = numbers.zfill(3)  # Ensure it's 3 digits (e.g., "002")
        
        all_results.append([draw_date, draw_time, numbers])

    params["page"] += 1

# Save to CSV with text formatting
with open("cash3_api_july.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Date", "Draw", "Winning_Numbers"])
    for row in all_results:
        writer.writerow([row[0], row[1], f"'{row[2]}"])  # Add single quote to keep Excel text format

print(f"✅ Saved {len(all_results)} records to cash3_api_july.csv (with leading zeros)")
