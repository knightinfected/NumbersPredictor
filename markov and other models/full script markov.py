import pandas as pd
import numpy as np

# Load and preprocess your CSV file
file_path =r"C:\Users\hmzmh\OneDrive\Desktop\scraper\cash3_2025_sorted.csv"
lottery_data = pd.read_csv(file_path)

# Cleaning data
lottery_data['Date'] = pd.to_datetime(lottery_data['Date'])
lottery_data['Winning_Numbers'] = lottery_data['Winning_Numbers'].str.replace("'", "").astype(int)

# Extract digits
lottery_data['Digit1'] = lottery_data['Winning_Numbers'] // 100
lottery_data['Digit2'] = (lottery_data['Winning_Numbers'] // 10) % 10
lottery_data['Digit3'] = lottery_data['Winning_Numbers'] % 10

# Sort data by date and draw order
draw_order = {'MIDDAY': 0, 'EVENING': 1, 'NIGHT': 2}
lottery_data['Draw_Order'] = lottery_data['Draw'].map(draw_order)
lottery_data.sort_values(by=['Date', 'Draw_Order'], inplace=True)

# Transition matrix function for Markov Chain
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

# Example prediction based on the latest known winning number (e.g., 084)
last_digits = [3, 7, 7]  # Replace with your last known digits

next_digit1 = predict_next_digit(last_digits[0], tm_digit1)
next_digit2 = predict_next_digit(last_digits[1], tm_digit2)
next_digit3 = predict_next_digit(last_digits[2], tm_digit3)

predicted_number = next_digit1 * 100 + next_digit2 * 10 + next_digit3

print(f"Predicted next winning number: {predicted_number:03d}")

input("Press Enter to exit...")
