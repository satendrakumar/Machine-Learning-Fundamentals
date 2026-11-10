"""Part 6 — Neural networks

A single neuron, the perceptron learning an AND gate, solving XOR
with a hidden layer, activation functions and counting parameters.
"""

import numpy as np


def step(z):
    return (z >= 0).astype(int)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


# --- A single neuron -------------------------------------------------------
x = np.array([2, 3])
w = np.array([0.5, -1])
b = 1
z = w @ x + b
print(f"Neuron: z = {z}, ReLU(z) = {relu(z)}  -> the neuron stays quiet")

# --- Perceptron learning the AND gate --------------------------------------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

w = np.zeros(2)
b = 0.0
eta = 0.1  # learning rate
for epoch in range(20):
    mistakes = 0
    for xi, yi in zip(X, y_and):
        y_hat = step(w @ xi + b)
        # Learning rule: nudge the weights toward the right answer.
        w += eta * (yi - y_hat) * xi
        b += eta * (yi - y_hat)
        mistakes += int(y_hat != yi)
    if mistakes == 0:
        break
print(f"\nPerceptron learned AND after {epoch + 1} epochs: w = {w}, b = {b:.1f}")
print("  outputs:", step(X @ w + b), " expected:", y_and)

# --- XOR needs a hidden layer -----------------------------------------------
# A single perceptron can't learn XOR. Two hidden neurons (OR and AND) can.
h1 = step(X[:, 0] + X[:, 1] - 0.5)  # OR
h2 = step(X[:, 0] + X[:, 1] - 1.5)  # AND
y_xor = step(h1 - h2 - 0.5)         # OR but not AND
print("\nXOR with a hidden layer:")
print(" x1 x2 | h1 h2 | y")
for (x1, x2), a, c, out in zip(X, h1, h2, y_xor):
    print(f"  {x1}  {x2} |  {a}  {c} | {out}")

# --- Activation functions -----------------------------------------------------
print("\nActivation functions:")
z = np.array([-5, -2, 0, 2, 5])
print(f"  z = {z}  sigmoid = {np.round(sigmoid(z), 3)}")
z = np.array([-2, -1, 0, 1, 2])
print(f"  z = {z}  tanh    = {np.round(np.tanh(z), 2)}")
z = np.array([-3, -1, 0, 2, 5])
print(f"  z = {z}  ReLU    = {relu(z)}")
print(f"  sigmoid's max slope is 0.25; over 5 layers: 0.25**5 = {0.25 ** 5:.4f} (vanishing gradient)")

# --- Counting parameters ------------------------------------------------------
def count_parameters(layer_sizes):
    total = 0
    for inputs, neurons in zip(layer_sizes, layer_sizes[1:]):
        total += inputs * neurons + neurons  # weights + biases
    return total


print("\nParameters:")
print(f"  3 -> 4 -> 2      : {count_parameters([3, 4, 2]):,}")
print(f"  784 -> 128 -> 10 : {count_parameters([784, 128, 10]):,}")
print(f"  10 -> 8 -> 4 -> 1 : {count_parameters([10, 8, 4, 1]):,}  (your-turn fraud detector)")
