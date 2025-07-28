readme_content = f"""
# Number Predictor using data from Cash 3 (Pick3) Lottery.

Number Predictor (000-999) using Cash3 Lottery data. Started out as a simple model (markov chain) into multi-model framework designed to predict Georgia Cash 3 lottery outcomes using a hybrid of statistical, machine learning, and signal processing techniques.
---
##  To run this, must have python installed. With the following dependencies 
- pandas numpy statsmodels scikit-learn scipy
##  Project Goals 
- One of these ideas popped up in my head and somehow ended up doing too much. This is done for fun or in the name of it.
- Explore whether short 3-digit lottery outcomes (000–999) follow *non-random* patterns. - Very annoying part
- Use real historical draw data to:
  - Train and evaluate predictive models.
  - Combine diverse techniques using weighted ensemble logic.
- Output top N numbers with the highest prediction confidence.

---

## 📂 Data Source
- CSV scraped from Georgia Lottery's public Cash 3 results via a custom API scraper.
- Structure: `Date, Draw, Winning_Numbers` - using this structure you could try this model on other data sources.
- Cleaned and sorted so that **most recent draw appears first**. - Pain in the butt 

---

##  Prediction Techniques Implemented (started out as markov chain only, will add the script here for that as well)

Each draw prediction includes the following models: 

###  Markov Chain
- Uses transitions from one number to the next.
- Works well for pattern recurrence.

###  Frequency Weighted
- Tracks digit frequency in recent draws.
- Prioritizes hot numbers.

###  ARIMA Trend Model
- Time series forecasting using `statsmodels`.
- Predicts future based on trend + seasonality.

###  Hot/Cold Analysis
- Compares recently hot numbers vs cold ones.
- Scores numbers based on deviation from average.

###  Sum & Parity Analysis
- Evaluates if digits tend to sum to specific ranges.
- Even/Odd (parity) and sum rules applied.

###  Difference Pattern
- Looks at the difference between digits and draw-to-draw.

###  Gap Analysis
- Time since last occurrence for each digit or combo.

###  Modulo Pattern
- Uses modular arithmetic to uncover cyclical behavior.

###  Entropy Trend
- Scores numbers by how "disordered" or "structured" they appear.

###  FFT Cycle Detection
- Fast Fourier Transform on digit frequency over time.
- Detects hidden periodic cycles.

### Higher-Order Markov
- Second-order transition logic to detect deeper sequential patterns.

###  Cluster Analysis
- Unsupervised ML clustering of draws using K-means.

###  Lagged Correlation
- Compares digits vs previous digits with temporal offset.

###  Machine Learning Predictor
- ML model trained on full feature set of previous draws using `RandomForest`.

###  Bayesian Updating
- Updates beliefs about what digits will appear based on past appearances.

---
## 📂 Some random Screenshots
<img width="1688" height="1136" alt="2025-07-28T15_46_27" src="https://github.com/user-attachments/assets/8a5d2c01-f967-438c-a4c3-d015c140134c" />
<img width="1602" height="1186" alt="2025-07-28T15_46_10" src="https://github.com/user-attachments/assets/ec196f97-0cac-40ef-a433-bdaa129c5efb" />
<img width="2226" height="1365" alt="2025-07-28T15_44_37" src="https://github.com/user-attachments/assets/827015cb-478b-4a48-97d9-189f05a972f4" />
##  Accuracy & Backtesting 
- Historical simulation from Jan 1 – Jul 25, 2025
- All predictions back-tested on actual outcomes
- Accuracy metrics include:
  - Exact hit
  - 2-digit match
  - 1-digit match
  - Total score
- Highest performing standalone model: **Gap Analysis** (Exact: 0.34%)

---

##  Ensemble Probability System
- Normalizes all models to [0, 1]
- Weights each method based on historical accuracy
- Scores all 000–999 combinations
- Outputs top **N = 10** predictions per draw

---

## 📁 Files
- `historical_predictions_2025.csv` – All daily predictions (per model)
- `historical_accuracy_2025.csv` – Summary of model performance
- `full_advanced.py` – Runs prediction models and ensemble logic
- `api scrape this year.py` – Pulls latest data from GA Lottery
- `README.md` – You’re here :)

---

##  Philosophy
While the lottery is widely considered random, this project operates under the hypothesis that:
- Small number sets (000–999) may reveal *non-random tendencies* over time.
- External predictors may capture hidden trends or operational quirks in the system.

---

---


# Save as a markdown file
readme_path = "/mnt/data/README.md"
with open(readme_path, "w") as f:
    f.write(readme_content)
