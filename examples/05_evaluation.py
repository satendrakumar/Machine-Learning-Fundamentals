"""Part 5 — Evaluating a classifier

The accuracy trap, confusion matrix, precision, recall, F1,
the threshold trade-off and a multiclass confusion matrix.
"""

import numpy as np

# --- The accuracy trap -------------------------------------------------------
# 1,000 card transactions, 10 are fraud (1). A lazy model says "genuine" (0) every time.
actual = np.array([1] * 10 + [0] * 990)
lazy_prediction = np.zeros(1000)
accuracy = np.mean(actual == lazy_prediction)
caught = np.sum((actual == 1) & (lazy_prediction == 1))
print(f"Lazy model: accuracy = {accuracy:.0%}, fraud caught = {caught} of 10")

# --- Confusion matrix from predictions --------------------------------------
def confusion(actual, predicted):
    tp = np.sum((predicted == 1) & (actual == 1))
    tn = np.sum((predicted == 0) & (actual == 0))
    fp = np.sum((predicted == 1) & (actual == 0))
    fn = np.sum((predicted == 0) & (actual == 1))
    return tp, tn, fp, fn


def metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1


# COVID test on 100 people: 10 sick, 90 healthy.
actual = np.array([1] * 10 + [0] * 90)
predicted = np.array([1] * 8 + [0] * 2      # sick: 8 caught, 2 missed
                     + [1] * 5 + [0] * 85)  # healthy: 5 false alarms, 85 cleared
tp, tn, fp, fn = confusion(actual, predicted)
print(f"\nCOVID test: TP={tp}  FN={fn}  FP={fp}  TN={tn}")

acc, prec, rec, f1 = metrics(tp, tn, fp, fn)
print(f"  Accuracy  = {acc:.1%}")
print(f"  Precision = {prec:.1%}   can I trust a 'yes'?")
print(f"  Recall    = {rec:.1%}   did we find all the sick people?")
print(f"  F1        = {f1:.1%}")

# Why F1 uses the harmonic mean, not the simple average.
p, r = 1.0, 0.01
print(f"\nP = 100%, R = 1%: average = {(p + r) / 2:.1%}, F1 = {2 * p * r / (p + r):.1%}")

# --- Precision-recall trade-off: move the threshold -------------------------
# Spam probabilities from a model, and whether each email really was spam.
probs = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05])
is_spam = np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 0])
print("\nThreshold  Precision  Recall")
for threshold in [0.3, 0.5, 0.8]:
    flagged = (probs >= threshold).astype(int)
    tp, tn, fp, fn = confusion(is_spam, flagged)
    _, prec, rec, _ = metrics(tp, tn, fp, fn)
    print(f"   {threshold:.1f}       {prec:5.0%}     {rec:5.0%}")

# --- Multiclass confusion matrix ---------------------------------------------
classes = ["Cat", "Dog", "Rabbit"]
# rows = actual, columns = predicted
cm = np.array([[8, 1, 1],
               [2, 7, 1],
               [0, 1, 9]])
print(f"\nMulticlass accuracy = {np.trace(cm)}/{cm.sum()} = {np.trace(cm) / cm.sum():.0%}")
precision = np.diag(cm) / cm.sum(axis=0)  # divide by column totals (predicted)
recall = np.diag(cm) / cm.sum(axis=1)     # divide by row totals (actual)
for c, p, r in zip(classes, precision, recall):
    print(f"  {c:6s} precision = {p:.0%}  recall = {r:.0%}")
print(f"  macro-average recall = {recall.mean():.0%}")
