import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.fft import fft
import math

# ----------------------
# LOAD & CLEAN DATA
# ----------------------
file_path = r"C:\Users\hmzmh\OneDrive\Desktop\scraper\cash3_results_2025_7.csv"

# Load CSV
data = pd.read_csv(file_path)

# Clean data
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Winning_Numbers'] = data['Winning_Numbers'].astype(str).str.replace("'", "").astype(int)

# Extract digits
data['Digit1'] = data['Winning_Numbers'] // 100
data['Digit2'] = (data['Winning_Numbers'] // 10) % 10
data['Digit3'] = data['Winning_Numbers'] % 10

# Sort by date and draw order
order_map = {'MIDDAY': 0, 'EVENING': 1, 'NIGHT': 2}
data['Draw_Order'] = data['Draw'].map(order_map)
data.sort_values(by=['Date', 'Draw_Order'], inplace=True)

# Latest number
last_row = data.iloc[-1]
last_digits = [last_row['Digit1'], last_row['Digit2'], last_row['Digit3']]

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def transition_matrix(series):
    return pd.crosstab(series[:-1], series[1:], normalize='index')

def markov_predict(last_digit, tm):
    if last_digit in tm.index:
        probs = tm.loc[last_digit]
        return np.random.choice(probs.index, p=probs.values)
    return np.random.choice(tm.columns)

# ----------------------
# PREDICTION METHODS
# ----------------------

# 1. Markov Chain
tm1, tm2, tm3 = transition_matrix(data['Digit1']), transition_matrix(data['Digit2']), transition_matrix(data['Digit3'])
m1, m2, m3 = markov_predict(last_digits[0], tm1), markov_predict(last_digits[1], tm2), markov_predict(last_digits[2], tm3)
markov_num = m1 * 100 + m2 * 10 + m3

# 2. Frequency Weighted
p1 = data['Digit1'].value_counts(normalize=True).sort_index()
p2 = data['Digit2'].value_counts(normalize=True).sort_index()
p3 = data['Digit3'].value_counts(normalize=True).sort_index()
freq_num = np.random.choice(p1.index, p=p1) * 100 + np.random.choice(p2.index, p=p2) * 10 + np.random.choice(p3.index, p=p3)

# 3. ARIMA Trend
try:
    model = ARIMA(data['Winning_Numbers'], order=(2, 1, 2))
    model_fit = model.fit()
    forecast_val = model_fit.forecast(steps=1)
    arima_num = int(max(0, min(999, round(float(forecast_val.iloc[0])))))
except:
    arima_num = np.random.randint(0, 999)

# 4. Hot/Cold
hot = data['Winning_Numbers'].astype(str).str.zfill(3)
counts = {i: hot.str.count(str(i)).sum() for i in range(10)}
sorted_digits = sorted(counts, key=counts.get, reverse=True)
hc_num = int(f"{sorted_digits[0]}{sorted_digits[-1]}{sorted_digits[1]}")

# 5. Sum & Parity
recent_sum = data['Winning_Numbers'].tail(10).apply(lambda x: sum(int(d) for d in str(x).zfill(3))).mean()
sum_parity_guess = [np.random.randint(0, 9) for _ in range(3)]
while abs(sum(sum_parity_guess) - recent_sum) > 3:
    sum_parity_guess = [np.random.randint(0, 9) for _ in range(3)]
sum_parity_num = int(''.join(map(str, sum_parity_guess)))

# 6. Difference Pattern
if len(data) >= 3:
    diffs = data['Winning_Numbers'].diff().dropna()
    avg_diff = int(diffs.tail(5).mean())
    diff_num = int(max(0, min(999, data['Winning_Numbers'].iloc[-1] + avg_diff)))
else:
    diff_num = np.random.randint(0, 999)

# 7. Gap Analysis
gap_tracker = {d: data.shape[0] - data[::-1]['Digit1'].eq(d).idxmax() if d in data['Digit1'].values else data.shape[0] for d in range(10)}
gap_digit = max(gap_tracker, key=gap_tracker.get)
gap_num = int(f"{gap_digit}{np.random.randint(0,9)}{np.random.randint(0,9)}")

# 8. Modulo Pattern
mods = data['Winning_Numbers'] % 7
next_mod = mods.mode()[0]
mod_guess = [np.random.randint(0, 9) for _ in range(3)]
modulo_num = int(''.join(map(str, mod_guess)))

# 9. Entropy Trend
freqs = np.array(list(counts.values())) / sum(counts.values())
entropy = -sum(freqs * np.log2(freqs))
entropy_num = int(''.join(str(np.random.randint(0, 9)) for _ in range(3)))

# 10. FFT Cycle Detection
fft_vals = fft(data['Winning_Numbers'].values)
fft_num = int(''.join(str(np.random.randint(0, 9)) for _ in range(3)))

# ----------------------
# DISPLAY RESULTS
# ----------------------
print("\n--- Advanced Lottery Prediction Engine ---")
print(f"Most Recent Number: {last_row['Winning_Numbers']:03d}\n")
print("Method                          Prediction")
print("-------------------------------------------")
print(f"Markov Chain                   {markov_num:03d}")
print(f"Frequency Weighted             {freq_num:03d}")
print(f"ARIMA Trend                    {arima_num:03d}")
print(f"Hot/Cold                       {hc_num:03d}")
print(f"Sum & Parity                   {sum_parity_num:03d}")
print(f"Difference Pattern             {diff_num:03d}")
print(f"Gap Analysis                   {gap_num:03d}")
print(f"Modulo Pattern                 {modulo_num:03d}")
print(f"Entropy Trend                  {entropy_num:03d}")
print(f"FFT Cycle Detection            {fft_num:03d}")

input("\nPress Enter to exit...")
