"""Part 4 — Classification

Logistic regression (sigmoid), binary vs multiclass (softmax)
vs multilabel, and categorical cross-entropy.
"""

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    e = np.exp(z)
    return e / e.sum()


# --- Binary: is this email spam? -------------------------------------------
w = np.array([1.5, 0.8])   # per "FREE", has a link
b = -2
email = np.array([2, 1])   # "FREE" appears 2 times, contains a link

z = w @ email + b
p = sigmoid(z)
print("Binary — spam or not?")
print(f"  z = {z:.1f},  p = sigmoid(z) = {p:.2f}")
print(f"  {p:.2f} >= 0.5  ->  {'Spam' if p >= 0.5 else 'Not spam'}")

# Same model on a few more emails — each lands on one side of the boundary.
emails = np.array([[0, 0], [1, 0], [1, 1], [3, 0]])
for e, pe in zip(emails, sigmoid(emails @ w + b)):
    print(f"  FREE x{e[0]}, link={e[1]}  ->  p = {pe:.2f}  {'Spam' if pe >= 0.5 else 'Not spam'}")

# --- Multiclass: softmax picks exactly one class ---------------------------
classes = ["Cat", "Dog", "Rabbit"]
scores = np.array([2.0, 1.0, 0.1])
probs = softmax(scores)
print("\nMulticlass — softmax:")
for c, s, pr in zip(classes, scores, probs):
    print(f"  {c:6s} score = {s:.1f}  e^z = {np.exp(s):5.2f}  p = {pr:.2f}")
print(f"  sum of probabilities = {probs.sum():.2f}  ->  prediction: {classes[np.argmax(probs)]}")

# Categorical cross-entropy: -log(probability of the true class)
print(f"  loss if the true answer is Cat: -log({probs[0]:.2f}) = {-np.log(probs[0]):.2f}")

# --- Multilabel: an independent sigmoid per label --------------------------
labels = ["Action", "Comedy", "Sci-Fi", "Romance"]
label_probs = np.array([0.92, 0.15, 0.81, 0.57])  # each from its own sigmoid
print('\nMultilabel — "Interstellar" (threshold 0.5):')
for name, pr in zip(labels, label_probs):
    print(f"  {name:8s} p = {pr:.2f}  {'yes' if pr >= 0.5 else 'no'}")
print(f"  sum = {label_probs.sum():.2f}  (doesn't need to be 1 — each label is decided on its own)")
