"""Part 2 — Regression

Linear regression y_hat = w*x + b, the worked examples from the slides,
learning w and b from data, and residuals.
"""

import numpy as np

# --- Worked example: one feature ---------------------------------------------
w, b = 0.05, 10  # lakh per sq ft, base price in lakh
area = 1000
print(f"1 feature : y_hat = {w} * {area} + {b} = {w * area + b:.0f} lakh")

# --- Worked example: several features ----------------------------------------
weights = np.array([0.05, 5, -0.8])  # area, bedrooms, age (negative: older = cheaper)
house = np.array([1200, 3, 10])
y_hat = weights @ house + b  # @ = multiply each pair and add them up
print(f"3 features: y_hat = {weights @ house:.0f} + {b} = {y_hat:.0f} lakh")

# --- Learning w and b from data ------------------------------------------------
rng = np.random.default_rng(42)
x = rng.uniform(500, 2000, size=20)                # area
y = 0.05 * x + 10 + rng.normal(0, 8, size=20)       # price with noise

w_learned, b_learned = np.polyfit(x, y, deg=1)      # least-squares best fit line
print(f"\nLearned from 20 houses: w = {w_learned:.4f}, b = {b_learned:.2f}")

# --- Residuals: actual minus predicted -----------------------------------------
predictions = w_learned * x + b_learned
residuals = y - predictions
print("\nFirst 5 houses:")
print(" area   actual  predicted  residual")
for xi, yi, pi, ri in zip(x[:5], y[:5], predictions[:5], residuals[:5]):
    print(f"{xi:5.0f}  {yi:7.1f}  {pi:9.1f}  {ri:+8.1f}")

# A loss function sums up all the residuals into one number (see part 3).
print(f"\nMean squared residual (MSE): {np.mean(residuals ** 2):.1f}")
