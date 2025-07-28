from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from datetime import datetime, timedelta

# Config
URL = "https://www.galottery.com/en-us/winning-numbers.html"
OUTPUT_FILE = "cash3_results.csv"

# Date range: last 2 months
end_date = datetime.today()
start_date = end_date - timedelta(days=60)
start_month = start_date.strftime("%B")
start_year = start_date.strftime("%Y")
end_month = end_date.strftime("%B")
end_year = end_date.strftime("%Y")

driver = webdriver.Chrome()
driver.get(URL)
wait = WebDriverWait(driver, 30)

def screenshot(step):
    driver.save_screenshot(f"{step}.png")

# ✅ Step 1: Click Advanced Search
adv_tab = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "Advanced Search")))
adv_tab.click()
time.sleep(2)
screenshot("1_advanced_tab")

# ✅ Step 2: Select Cash 3
dropdown = wait.until(EC.presence_of_element_located((By.ID, "advSearchGameSelect")))
Select(dropdown).select_by_visible_text("Cash 3")
screenshot("2_selected_cash3")

# ✅ Step 3: Select date range
# ✅ Correct dropdown IDs for Advanced Search
Select(driver.find_element(By.ID, "advSearchbyMonth")).select_by_visible_text("July")
Select(driver.find_element(By.ID, "advSearchbyYear")).select_by_visible_text("2025")


# ✅ Step 4: Click Submit (correct button by ID)
submit_btn = wait.until(EC.element_to_be_clickable((By.ID, "btnSearchByMonth")))
submit_btn.click()
time.sleep(3)
screenshot("4_after_submit")

# ✅ Step 5: Wait for results table
wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.table")))

# ✅ Step 6: Scrape data from multiple pages
data = []
while True:
    rows = driver.find_elements(By.CSS_SELECTOR, "table.table tbody tr")
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) >= 3:
            date = cols[0].text.strip()
            draw = cols[1].text.strip()
            # Extract numbers properly
            numbers_container = cols[2].find_elements(By.CSS_SELECTOR, "div.lotto-numbers-list span i")
            numbers = "".join([n.text for n in numbers_container])
            numbers = numbers.zfill(3)  # ensures 3 digits (e.g., 2 → 002)
            data.append([date, draw, numbers])

    # ✅ Try to click next page
    try:
        next_btn = driver.find_element(By.CSS_SELECTOR, "a[aria-label='Go to next page']")
        if "disabled" in next_btn.get_attribute("class"):
            break
        next_btn.click()
        time.sleep(2)
        screenshot("page_next")
    except:
        break
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.lotto-numbers-list")))

driver.quit()

# ✅ Save results
df = pd.DataFrame(data, columns=["Date", "Draw", "Winning_Numbers"])
df.to_csv(OUTPUT_FILE, index=False)
print(f"Scraped {len(data)} records → saved to {OUTPUT_FILE}")
