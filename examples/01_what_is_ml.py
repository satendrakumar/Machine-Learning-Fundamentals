"""Part 1 — What is machine learning?

Features vs labels, "learning the rule" instead of writing it,
and splitting data into a training set and a test set.
"""

import numpy as np

# --- Features (x) and labels (y) -------------------------------------------
# Same house table as the slide. Each row is one example.
#            area  bedrooms  age
X = np.array([[1000, 2, 5],
              [1500, 3, 2],
              [800, 1, 15]])
y = np.array([55, 90, 35])  # price in ₹ lakh — the label we want to predict

print("Features (area, bedrooms, age):")
print(X)
print("Labels (price, ₹ lakh):", y)

# --- Traditional programming vs machine learning ---------------------------
# Traditional: a human writes the rule.
def rule_written_by_human(area):
    return 0.05 * area + 10

# Machine learning: we give examples + answers, the computer finds the rule.
# Here the hidden "true" rule is price = 0.06 * area + 5, plus some noise.
rng = np.random.default_rng(0)
area = rng.uniform(500, 2000, size=100)
price = 0.06 * area + 5 + rng.normal(0, 3, size=100)

w, b = np.polyfit(area, price, deg=1)  # "training": find the best straight line
print(f"\nRule written by a human : price = 0.05 * area + 10")
print(f"Rule learned from data  : price = {w:.3f} * area + {b:.2f}")

# --- Train / test split (80% / 20%) ----------------------------------------
# Practice questions vs the final exam: we evaluate on data the model never saw.
idx = rng.permutation(len(area))
train, test = idx[:80], idx[80:]

w, b = np.polyfit(area[train], price[train], deg=1)
test_pred = w * area[test] + b
test_error = np.mean(np.abs(price[test] - test_pred))
print(f"\nTrained on {len(train)} houses, tested on {len(test)} unseen houses")
print(f"Average error on the test set: ₹{test_error:.2f} lakh")
