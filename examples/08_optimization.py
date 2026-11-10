"""Part 8 — Optimization

Gradient descent, the effect of the learning rate, batch vs mini-batch,
and SGD vs Momentum vs RMSProp vs Adam on the same problem.
"""

import numpy as np

# --- Gradient descent on a simple bowl: L(w) = (w - 3)^2 ---------------------
def loss(w):
    return (w - 3) ** 2


def gradient(w):
    return 2 * (w - 3)


print("Gradient descent on L(w) = (w - 3)^2, starting at w = 0:")
for eta in [0.01, 0.1, 1.1]:
    w = 0.0
    for _ in range(20):
        w = w - eta * gradient(w)
    verdict = {0.01: "too small: barely moved", 0.1: "just right",
               1.1: "too big: overshoots and explodes"}[eta]
    print(f"  learning rate {eta:<4}  after 20 steps w = {w:10.3f}  loss = {loss(w):12.3f}  ({verdict})")

# --- Batch GD vs mini-batch SGD for linear regression ---------------------
rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, size=1000)
y = 4 * x + 2 + rng.normal(0, 0.1, size=1000)   # true w = 4, b = 2


def gradients(w, b, xb, yb):
    error = (w * xb + b) - yb
    return 2 * np.mean(error * xb), 2 * np.mean(error)


print("\nLinear regression (true w = 4, b = 2), 1 epoch = one pass over 1000 examples:")
for batch_size in [1000, 32]:
    w, b = 0.0, 0.0
    updates = 0
    for epoch in range(5):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            batch = order[start:start + batch_size]
            dw, db = gradients(w, b, x[batch], y[batch])
            w, b = w - 0.1 * dw, b - 0.1 * db
            updates += 1
    name = "Batch GD" if batch_size == len(x) else f"Mini-batch ({batch_size})"
    print(f"  {name:16s} 5 epochs, {updates:3d} updates -> w = {w:.2f}, b = {b:.2f}")

# --- Optimizers compared on a stretched bowl -------------------------------
# L(w1, w2) = w1^2 + 100 * w2^2 is steep in w2 and flat in w1. Plain SGD must use a
# tiny learning rate (or it zig-zags across the steep direction), so it crawls along w1.
def loss2(w):
    return w[0] ** 2 + 100 * w[1] ** 2


def grad2(w):
    return np.array([2 * w[0], 200 * w[1]])


def steps_to_converge(optimizer, eta, max_steps=2000):
    w = np.array([-4.0, 1.0])
    state = {}
    for t in range(1, max_steps + 1):
        w = optimizer(w, grad2(w), state, t, eta)
        if loss2(w) < 0.001:
            return f"{t} steps"
    return f"not there after {max_steps} steps (loss = {loss2(w):.3g})"


def sgd(w, g, state, t, eta):
    return w - eta * g


def momentum(w, g, state, t, eta, beta=0.9):
    state["v"] = beta * state.get("v", 0) + (1 - beta) * g
    return w - eta * state["v"]


def rmsprop(w, g, state, t, eta, beta=0.9, eps=1e-8):
    state["s"] = beta * state.get("s", 0) + (1 - beta) * g ** 2
    return w - eta / (np.sqrt(state["s"]) + eps) * g


def adam(w, g, state, t, eta, beta1=0.9, beta2=0.999, eps=1e-8):
    state["m"] = beta1 * state.get("m", 0) + (1 - beta1) * g
    state["v"] = beta2 * state.get("v", 0) + (1 - beta2) * g ** 2
    m_hat = state["m"] / (1 - beta1 ** t)   # bias correction
    v_hat = state["v"] / (1 - beta2 ** t)
    return w - eta * m_hat / (np.sqrt(v_hat) + eps)


print("\nSteps to reach loss < 0.001 on a stretched bowl (start loss = 116):")
for name, opt, eta in [("SGD", sgd, 0.009), ("SGD", sgd, 0.01),
                       ("Momentum", momentum, 0.03),
                       ("RMSProp", rmsprop, 0.1), ("Adam", adam, 0.1)]:
    print(f"  {name:9s} lr = {eta:<6} {steps_to_converge(opt, eta)}")
