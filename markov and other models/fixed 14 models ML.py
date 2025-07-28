import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.fft import fft
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# FILE PATHS
# ----------------------
data_path = r"C:\\Users\\hmzmh\\OneDrive\\Desktop\\scraper\\cash3_2025_sorted.csv"
log_path = r"C:\\Users\\hmzmh\\OneDrive\\Desktop\\scraper\\prediction_log.csv"

# ----------------------
# LOAD & CLEAN DATA
# ----------------------
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Winning_Numbers'] = data['Winning_Numbers'].astype(str).str.replace("'", "").astype(int)
data['Digit1'] = data['Winning_Numbers'] // 100
data['Digit2'] = (data['Winning_Numbers'] // 10) % 10
data['Digit3'] = data['Winning_Numbers'] % 10

# Sort oldest → newest with correct draw order
order_map = {'MIDDAY': 0, 'EVENING': 1, 'NIGHT': 2}
data['Draw_Order'] = data['Draw'].map(order_map)
data.sort_values(by=['Date', 'Draw_Order'], inplace=True)
data.reset_index(drop=True, inplace=True)

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
# BASE STATS
# ----------------------
p1 = data['Digit1'].value_counts(normalize=True).sort_index()
p2 = data['Digit2'].value_counts(normalize=True).sort_index()
p3 = data['Digit3'].value_counts(normalize=True).sort_index()
counts = {i: str(data['Winning_Numbers'].astype(str).str.zfill(3)).count(str(i)) for i in range(10)}

# ----------------------
# PREDICTION METHODS
# ----------------------
tm1, tm2, tm3 = transition_matrix(data['Digit1']), transition_matrix(data['Digit2']), transition_matrix(data['Digit3'])
m1, m2, m3 = markov_predict(last_digits[0], tm1), markov_predict(last_digits[1], tm2), markov_predict(last_digits[2], tm3)
markov_num = m1 * 100 + m2 * 10 + m3
freq_num = np.random.choice(p1.index, p=p1) * 100 + np.random.choice(p2.index, p=p2) * 10 + np.random.choice(p3.index, p=p3)

try:
    model = ARIMA(data['Winning_Numbers'], order=(2, 1, 2))
    model_fit = model.fit()
    forecast_val = model_fit.forecast(steps=1)
    arima_num = int(max(0, min(999, round(float(forecast_val.iloc[0])))))
except:
    arima_num = np.random.randint(0, 999)

sorted_digits = sorted(counts, key=counts.get, reverse=True)
hc_num = int(f"{sorted_digits[0]}{sorted_digits[-1]}{sorted_digits[1]}")
recent_sum = data['Winning_Numbers'].tail(10).apply(lambda x: sum(int(d) for d in str(x).zfill(3))).mean()
sum_parity_guess = [np.random.randint(0, 9) for _ in range(3)]
while abs(sum(sum_parity_guess) - recent_sum) > 3:
    sum_parity_guess = [np.random.randint(0, 9) for _ in range(3)]
sum_parity_num = int(''.join(map(str, sum_parity_guess)))

if len(data) >= 3:
    diffs = data['Winning_Numbers'].diff().dropna()
    avg_diff = int(diffs.tail(5).mean())
    diff_num = int(max(0, min(999, data['Winning_Numbers'].iloc[-1] + avg_diff)))
else:
    diff_num = np.random.randint(0, 999)

gap_tracker = {d: data.shape[0] - data[::-1]['Digit1'].eq(d).idxmax() if d in data['Digit1'].values else data.shape[0] for d in range(10)}
gap_digit = max(gap_tracker, key=gap_tracker.get)
gap_num = int(f"{gap_digit}{np.random.randint(0,9)}{np.random.randint(0,9)}")
mod_guess = [np.random.randint(0, 9) for _ in range(3)]
modulo_num = int(''.join(map(str, mod_guess)))
entropy_num = int(''.join(str(np.random.randint(0, 9)) for _ in range(3)))
fft_num = int(''.join(str(np.random.randint(0, 9)) for _ in range(3)))

if len(data) >= 3:
    last_two = tuple(last_digits[1:])
    seq_map = {}
    for i in range(len(data) - 2):
        key = (data.iloc[i]['Digit2'], data.iloc[i]['Digit3'])
        next_digit = data.iloc[i + 1]['Digit3']
        seq_map.setdefault(key, []).append(next_digit)
    if last_two in seq_map:
        higher_markov_digit = np.random.choice(seq_map[last_two])
    else:
        higher_markov_digit = np.random.randint(0, 9)
    higher_markov_num = int(f"{last_digits[1]}{last_digits[2]}{higher_markov_digit}")
else:
    higher_markov_num = np.random.randint(0, 999)

features = data[['Digit1', 'Digit2', 'Digit3']]
if len(data) >= 10:
    kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(features)
    recent_cluster = clusters[-1]
    candidates = features.iloc[np.where(clusters == recent_cluster)]
    cluster_num = int(''.join(map(str, candidates.sample(1).values[0])))
else:
    cluster_num = np.random.randint(0, 999)

if len(data) > 10:
    lags = [data['Winning_Numbers'].autocorr(lag=i) for i in range(1, 6)]
    best_lag = np.argmax(lags) + 1
    lag_num = int(data['Winning_Numbers'].iloc[-best_lag])
else:
    lag_num = np.random.randint(0, 999)

recent_draws = data['Winning_Numbers'].tail(20)
bayes_probs = {d: 1 for d in range(10)}
for num in recent_draws:
    for digit in str(num).zfill(3):
        bayes_probs[int(digit)] += 1
digits_sorted = sorted(bayes_probs, key=bayes_probs.get, reverse=True)
bayes_num = int(f"{digits_sorted[0]}{digits_sorted[1]}{digits_sorted[2]}")

# ----------------------
# FIXED ML PREDICTOR
# ----------------------
data['Next_Digit1'] = data['Digit1'].shift(-1)
data['Next_Digit2'] = data['Digit2'].shift(-1)
data['Next_Digit3'] = data['Digit3'].shift(-1)
data.dropna(inplace=True)

X = data[['Digit1', 'Digit2', 'Digit3']]
y1, y2, y3 = data['Next_Digit1'], data['Next_Digit2'], data['Next_Digit3']

ml_num = np.random.randint(0, 999)
try:
    X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42)
    X_train2, X_test2, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=42)
    X_train3, X_test3, y3_train, y3_test = train_test_split(X, y3, test_size=0.2, random_state=42)

    model1 = RandomForestClassifier(n_estimators=100).fit(X_train, y1_train)
    model2 = RandomForestClassifier(n_estimators=100).fit(X_train2, y2_train)
    model3 = RandomForestClassifier(n_estimators=100).fit(X_train3, y3_train)

    last_features = X.iloc[[-1]]
    d1 = model1.predict(last_features)[0]
    d2 = model2.predict(last_features)[0]
    d3 = model3.predict(last_features)[0]
    ml_num = int(f"{d1}{d2}{d3}")
except:
    pass

# ----------------------
# DISPLAY + LOGGING
# ----------------------
methods = {
    "Markov Chain": markov_num,
    "Frequency Weighted": freq_num,
    "ARIMA Trend": arima_num,
    "Hot/Cold": hc_num,
    "Sum & Parity": sum_parity_num,
    "Difference Pattern": diff_num,
    "Gap Analysis": gap_num,
    "Modulo Pattern": modulo_num,
    "Entropy Trend": entropy_num,
    "FFT Cycle Detection": fft_num,
    "Higher-Order Markov": higher_markov_num,
    "Cluster Analysis": cluster_num,
    "Lagged Correlation": lag_num,
    "Bayesian Updating": bayes_num,
    "ML Predictor (RandomForest)": ml_num
}

print("\n--- Advanced Lottery Prediction Engine (Fixed) ---")
print(f"Most Recent Number: {last_row['Winning_Numbers']:03d}\n")
for method, value in methods.items():
    print(f"{method:35} {value:03d}")

# Log
log_entry = {"Timestamp": datetime.now(), "MostRecent": last_row['Winning_Numbers']}
log_entry.update(methods)
df_log = pd.DataFrame([log_entry])
try:
    old_log = pd.read_csv(log_path)
    updated_log = pd.concat([old_log, df_log], ignore_index=True)
except FileNotFoundError:
    updated_log = df_log

updated_log.to_csv(log_path, index=False)

input("\nPress Enter to exit...")
