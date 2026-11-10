"""Part 7 — Feedforward & backpropagation

The single-neuron worked example from the slides (forward pass,
chain rule, update), then a small network that learns XOR by itself.
"""

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --- Worked example: one sigmoid neuron ------------------------------------
x = np.array([1.0, 2.0])
w = np.array([0.5, -0.3])
b = 0.1
y = 1
eta = 0.1

# Forward pass
z = w @ x + b
y_hat = sigmoid(z)
loss = (y - y_hat) ** 2
print(f"Forward : z = {z:.2f}, y_hat = {y_hat:.3f}, loss = {loss:.3f}")

# Backward pass (chain rule): dL/dw = dL/dy_hat * dy_hat/dz * dz/dw
dL_dyhat = -2 * (y - y_hat)
dyhat_dz = y_hat * (1 - y_hat)
dL_dw = dL_dyhat * dyhat_dz * x   # dz/dw = x
dL_db = dL_dyhat * dyhat_dz * 1   # dz/db = 1
print(f"Gradients: dL/dw1 = {dL_dw[0]}, dL/dw2 = {dL_dw[1]}, dL/db = {dL_db}")

# Update: step against the gradient
w = w - eta * dL_dw
b = b - eta * dL_db
print(f"Updated : w1 = {w[0]:.3f}, w2 = {w[1]:.3f}, b = {b:.3f}")

z = w @ x + b
y_hat = sigmoid(z)
print(f"Forward again: z = {z:.2f}, y_hat = {y_hat:.3f}, loss = {(y - y_hat) ** 2:.3f}  (was 0.25)")

# --- A 2 -> 4 -> 1 network learns XOR ---------------------------------------
# Same recipe, repeated thousands of times: predict -> loss -> gradients -> update.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)

rng = np.random.default_rng(1)
W1 = rng.normal(0, 1, size=(2, 4))
b1 = np.zeros(4)
W2 = rng.normal(0, 1, size=(4, 1))
b2 = np.zeros(1)
eta = 1.0

print("\nTraining a 2 -> 4 -> 1 network on XOR:")
for epoch in range(5001):
    # Feedforward
    H = sigmoid(X @ W1 + b1)       # hidden layer
    Y_hat = sigmoid(H @ W2 + b2)   # output layer
    loss = np.mean((Y - Y_hat) ** 2)

    # Backpropagation: start at the loss and work backwards
    dZ2 = -2 * (Y - Y_hat) / len(X) * Y_hat * (1 - Y_hat)
    dW2 = H.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dZ1 = (dZ2 @ W2.T) * H * (1 - H)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    # Gradient descent update
    W2 -= eta * dW2
    b2 -= eta * db2
    W1 -= eta * dW1
    b1 -= eta * db1

    if epoch % 1000 == 0:
        print(f"  epoch {epoch:4d}  loss = {loss:.4f}")

print("Predictions:")
for xi, pi in zip(X, Y_hat):
    print(f"  {xi.astype(int)} -> {pi[0]:.2f}  (rounded: {round(pi[0])})")
