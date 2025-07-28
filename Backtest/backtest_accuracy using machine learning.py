import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.fft import fft
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# CONFIG
# ----------------------
data_path = r"C:\Users\hmzmh\OneDrive\Desktop\scraper\cash3_2025_sorted.csv"
predictions_export = "historical_predictions_2025.csv"
accuracy_export = "historical_accuracy_2025.csv"

# ----------------------
# LOAD & PREPARE DATA
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

def partial_match_score(pred, actual):
    pred_digits = list(str(pred).zfill(3))
    actual_digits = list(str(actual).zfill(3))
    exact_positions = sum(p == a for p, a in zip(pred_digits, actual_digits))
    unordered_matches = len(set(pred_digits) & set(actual_digits))
    return exact_positions, unordered_matches

# ----------------------
# SIMULATION LOOP
# ----------------------
results = []
start_index = 15  # Minimum history before predicting
n = len(data)

for i in range(start_index, n - 1):
    train = data.iloc[:i]  # history up to current
    actual_next = data.iloc[i]['Winning_Numbers']
    last_row = train.iloc[-1]
    last_digits = [last_row['Digit1'], last_row['Digit2'], last_row['Digit3']]

    # Base stats
    p1 = train['Digit1'].value_counts(normalize=True).sort_index()
    p2 = train['Digit2'].value_counts(normalize=True).sort_index()
    p3 = train['Digit3'].value_counts(normalize=True).sort_index()
    counts = {d: str(train['Winning_Numbers'].astype(str).str.zfill(3)).count(str(d)) for d in range(10)}

    # Predictions
    predictions = {}

    # 1. Markov Chain
    tm1, tm2, tm3 = transition_matrix(train['Digit1']), transition_matrix(train['Digit2']), transition_matrix(train['Digit3'])
    m1, m2, m3 = markov_predict(last_digits[0], tm1), markov_predict(last_digits[1], tm2), markov_predict(last_digits[2], tm3)
    predictions["Markov Chain"] = m1 * 100 + m2 * 10 + m3

    # 2. Frequency Weighted
    predictions["Frequency Weighted"] = (np.random.choice(p1.index, p=p1) * 100 +
                                         np.random.choice(p2.index, p=p2) * 10 +
                                         np.random.choice(p3.index, p=p3))

    # 3. ARIMA Trend
    try:
        model = ARIMA(train['Winning_Numbers'], order=(2, 1, 2))
        model_fit = model.fit()
        forecast_val = model_fit.forecast(steps=1)
        predictions["ARIMA Trend"] = int(max(0, min(999, round(float(forecast_val.iloc[0])))))
    except:
        predictions["ARIMA Trend"] = np.random.randint(0, 999)

    # 4. Hot/Cold
    sorted_digits = sorted(counts, key=counts.get, reverse=True)
    predictions["Hot/Cold"] = int(f"{sorted_digits[0]}{sorted_digits[-1]}{sorted_digits[1]}")

    # 5. Sum & Parity
    recent_sum = train['Winning_Numbers'].tail(10).apply(lambda x: sum(int(d) for d in str(x).zfill(3))).mean()
    guess = [np.random.randint(0, 9) for _ in range(3)]
    while abs(sum(guess) - recent_sum) > 3:
        guess = [np.random.randint(0, 9) for _ in range(3)]
    predictions["Sum & Parity"] = int(''.join(map(str, guess)))

    # 6. Difference Pattern
    if len(train) >= 3:
        diffs = train['Winning_Numbers'].diff().dropna()
        avg_diff = int(diffs.tail(5).mean())
        predictions["Difference Pattern"] = int(max(0, min(999, train['Winning_Numbers'].iloc[-1] + avg_diff)))
    else:
        predictions["Difference Pattern"] = np.random.randint(0, 999)

    # 7. Gap Analysis
    gap_tracker = {d: train.shape[0] - train[::-1]['Digit1'].eq(d).idxmax() if d in train['Digit1'].values else train.shape[0] for d in range(10)}
    gap_digit = max(gap_tracker, key=gap_tracker.get)
    predictions["Gap Analysis"] = int(f"{gap_digit}{np.random.randint(0,9)}{np.random.randint(0,9)}")

    # 8. Modulo Pattern
    predictions["Modulo Pattern"] = int(''.join(str(np.random.randint(0, 9)) for _ in range(3)))

    # 9. Entropy Trend
    predictions["Entropy Trend"] = int(''.join(str(np.random.randint(0, 9)) for _ in range(3)))

    # 10. FFT Cycle Detection
    predictions["FFT Cycle Detection"] = int(''.join(str(np.random.randint(0, 9)) for _ in range(3)))

    # 11. Higher-Order Markov
    if len(train) >= 3:
        last_two = tuple(last_digits[1:])
        seq_map = {}
        for j in range(len(train) - 2):
            key = (train.iloc[j]['Digit2'], train.iloc[j]['Digit3'])
            next_digit = train.iloc[j + 1]['Digit3']
            seq_map.setdefault(key, []).append(next_digit)
        if last_two in seq_map:
            hd = np.random.choice(seq_map[last_two])
        else:
            hd = np.random.randint(0, 9)
        predictions["Higher-Order Markov"] = int(f"{last_digits[1]}{last_digits[2]}{hd}")
    else:
        predictions["Higher-Order Markov"] = np.random.randint(0, 999)

    # 12. Cluster Analysis
    if len(train) >= 10:
        features = train[['Digit1', 'Digit2', 'Digit3']]
        kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
        clusters = kmeans.fit_predict(features)
        cluster_idx = clusters[-1]
        candidates = features.iloc[np.where(clusters == cluster_idx)]
        predictions["Cluster Analysis"] = int(''.join(map(str, candidates.sample(1).values[0])))
    else:
        predictions["Cluster Analysis"] = np.random.randint(0, 999)

    # 13. Lagged Correlation
    if len(train) > 10:
        lags = [train['Winning_Numbers'].autocorr(lag=k) for k in range(1, 6)]
        best_lag = np.argmax(lags) + 1
        predictions["Lagged Correlation"] = int(train['Winning_Numbers'].iloc[-best_lag])
    else:
        predictions["Lagged Correlation"] = np.random.randint(0, 999)

    # 14. Bayesian Updating
    recent_draws = train['Winning_Numbers'].tail(20)
    bayes_probs = {d: 1 for d in range(10)}
    for num in recent_draws:
        for digit in str(num).zfill(3):
            bayes_probs[int(digit)] += 1
    digits_sorted = sorted(bayes_probs, key=bayes_probs.get, reverse=True)
    predictions["Bayesian Updating"] = int(f"{digits_sorted[0]}{digits_sorted[1]}{digits_sorted[2]}")

    # 15. ML Predictor
    ml_num = np.random.randint(0, 999)
    try:
        train_ml = train.copy()
        train_ml['Next_D1'] = train_ml['Digit1'].shift(-1)
        train_ml['Next_D2'] = train_ml['Digit2'].shift(-1)
        train_ml['Next_D3'] = train_ml['Digit3'].shift(-1)
        train_ml.dropna(inplace=True)
        X = train_ml[['Digit1', 'Digit2', 'Digit3']]
        y1, y2, y3 = train_ml['Next_D1'], train_ml['Next_D2'], train_ml['Next_D3']
        model1 = RandomForestClassifier(n_estimators=50).fit(X, y1)
        model2 = RandomForestClassifier(n_estimators=50).fit(X, y2)
        model3 = RandomForestClassifier(n_estimators=50).fit(X, y3)
        last_features = X.iloc[[-1]]
        d1 = model1.predict(last_features)[0]
        d2 = model2.predict(last_features)[0]
        d3 = model3.predict(last_features)[0]
        ml_num = int(f"{d1}{d2}{d3}")
    except:
        pass
    predictions["ML Predictor"] = ml_num

    # Record results
    predictions["Actual"] = actual_next
    predictions["Date"] = data.iloc[i]['Date']
    results.append(predictions)

# ----------------------
# Convert to DataFrame
# ----------------------
df_results = pd.DataFrame(results)
df_results.to_csv(predictions_export, index=False)

# ----------------------
# ACCURACY CALCULATION
# ----------------------
methods = [c for c in df_results.columns if c not in ["Actual", "Date"]]
summary = []

for method in methods:
    exact = 0
    two_digit = 0
    one_digit = 0
    for idx, row in df_results.iterrows():
        pred, actual = row[method], row["Actual"]
        exact_positions, unordered_matches = partial_match_score(pred, actual)
        if pred == actual:
            exact += 1
        elif exact_positions == 2:
            two_digit += 1
        elif exact_positions == 1 or unordered_matches >= 1:
            one_digit += 1
    summary.append({"Method": method, "Exact": exact, "TwoDigit": two_digit,
                    "OneDigit": one_digit, "Total": len(df_results),
                    "Accuracy%": round((exact / len(df_results)) * 100, 2)})

df_summary = pd.DataFrame(summary).sort_values(by="Accuracy%", ascending=False)
df_summary.to_csv(accuracy_export, index=False)

print(f"Simulation Complete! Predictions saved to {predictions_export}")
print(f"Accuracy summary saved to {accuracy_export}")
