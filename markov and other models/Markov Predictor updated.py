import pandas as pd
import numpy as np

# Correct path to your CSV file
file_path = r"C:\Users\hmzmh\OneDrive\Desktop\scraper\cash3_2025_sorted.csv"

# Load data
lottery_data = pd.read_csv(file_path)

# Clean data
lottery_data['Date'] = pd.to_datetime(lottery_data['Date'])
lottery_data['Winning_Numbers'] = lottery_data['Winning_Numbers'].astype(str).str.replace("'", "").astype(int)

# Extract digits
lottery_data['Digit1'] = lottery_data['Winning_Numbers'] // 100
lottery_data['Digit2'] = (lottery_data['Winning_Numbers'] // 10) % 10
lottery_data['Digit3'] = lottery_data['Winning_Numbers'] % 10

# Sort by date and draw order
draw_order = {'MIDDAY': 0, 'EVENING': 1, 'NIGHT': 2}
lottery_data['Draw_Order'] = lottery_data['Draw'].map(draw_order)
lottery_data.sort_values(by=['Date', 'Draw_Order'], inplace=True)

# Get the most recent number automatically
last_row = lottery_data.iloc[-1]
last_digits = [last_row['Digit1'], last_row['Digit2'], last_row['Digit3']]

# Transition matrix function (Markov Chain)
def transition_matrix(series):
    return pd.crosstab(series[:-1], series[1:], normalize='index')

# Create transition matrices
tm_digit1 = transition_matrix(lottery_data['Digit1'])
tm_digit2 = transition_matrix(lottery_data['Digit2'])
tm_digit3 = transition_matrix(lottery_data['Digit3'])

# Prediction function
def predict_next_digit(last_digit, transition_matrix):
    if last_digit in transition_matrix.index:
        probabilities = transition_matrix.loc[last_digit]
        return np.random.choice(probabilities.index, p=probabilities.values)
    else:
        return np.random.choice(transition_matrix.columns)

# Predict next digits using Markov Chain
next_digit1 = predict_next_digit(last_digits[0], tm_digit1)
next_digit2 = predict_next_digit(last_digits[1], tm_digit2)
next_digit3 = predict_next_digit(last_digits[2], tm_digit3)

markov_prediction = next_digit1 * 100 + next_digit2 * 10 + next_digit3

# Alternative prediction using frequency-based sampling
digit1_probs = lottery_data['Digit1'].value_counts(normalize=True).sort_index()
digit2_probs = lottery_data['Digit2'].value_counts(normalize=True).sort_index()
digit3_probs = lottery_data['Digit3'].value_counts(normalize=True).sort_index()

alt_digit1 = np.random.choice(digit1_probs.index, p=digit1_probs)
alt_digit2 = np.random.choice(digit2_probs.index, p=digit2_probs)
alt_digit3 = np.random.choice(digit3_probs.index, p=digit3_probs)

freq_prediction = alt_digit1 * 100 + alt_digit2 * 10 + alt_digit3

print("\n--- Predictions ---")
print(f"Most recent number: {last_row['Winning_Numbers']:03d}")
print(f"Markov Chain Prediction: {markov_prediction:03d}")
print(f"Frequency-based Prediction: {freq_prediction:03d}")

input("\nPress Enter to exit...")
