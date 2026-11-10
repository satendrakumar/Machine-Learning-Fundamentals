"""Part 3 — Loss functions

MSE, MAE, RMSE, the loss landscape and binary cross-entropy,
using the same numbers as the slides.
"""

import numpy as np


def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)


def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))


def rmse(y, y_hat):
    return np.sqrt(mse(y, y_hat))


def binary_cross_entropy(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# --- MSE on the house table ------------------------------------------------
actual = np.array([200, 300, 400])
predicted = np.array([210, 280, 400])
print("Errors:", actual - predicted)
print(f"MSE  = {mse(actual, predicted):.1f}")
print(f"RMSE = {rmse(actual, predicted):.1f}  (back in lakh — 'off by about ₹12.9 lakh')")

# --- MSE vs MAE with one outlier --------------------------------------------
errors = np.array([1, 1, 1, 10])
zeros = np.zeros(4)
print(f"\nErrors {errors.tolist()}:")
print(f"MAE = {mae(errors, zeros):.2f}   (treats the outlier linearly)")
print(f"MSE = {mse(errors, zeros):.2f}  (the 10 becomes 100 when squared)")

# --- The loss landscape: try many values of w ------------------------------
# Data that follows y = 2x exactly, so the best w is 2.
x = np.array([1, 2, 3, 4])
y = 2 * x
print("\nLoss landscape (MSE for different w) — a bowl with its bottom at w = 2:")
for w in np.arange(0, 4.5, 0.5):
    loss = mse(y, w * x)
    print(f"  w = {w:3.1f}  loss = {loss:6.2f}  {'#' * int(loss)}")

# --- Binary cross-entropy (true answer is "yes", y = 1) --------------------
print("\nBinary cross-entropy when the true answer is yes:")
for p in [0.9, 0.5, 0.1]:
    print(f"  model says p(yes) = {p}  ->  loss = {binary_cross_entropy(1, p):.2f}")
