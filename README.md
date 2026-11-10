# Machine Learning Fundamentals

A beginner-friendly slide deck covering:

1. What machine learning is (supervised / unsupervised / reinforcement, features & labels, workflow)
2. Regression (linear regression, worked examples, residuals)
3. Loss functions (MSE, MAE, RMSE, binary & categorical cross-entropy)
4. Classification (logistic regression & sigmoid, binary vs multiclass/softmax vs multilabel)
5. Evaluation (confusion matrix, accuracy, precision, recall, F1, trade-offs)
6. Neural networks (neuron, perceptron, XOR, hidden layers, activation functions — sigmoid, tanh, ReLU, softmax — parameter counting, hyperparameters, tuning)
7. Feedforward & backpropagation (chain rule with a fully worked numeric example, choosing a loss)
8. Optimization (gradient descent, learning rate, SGD / mini-batch, Momentum, RMSProp, Adam)

## Viewing the presentation

```bash
open presentation/ml_fundamentals.html
```

It's a single HTML file built with [reveal.js](https://revealjs.com) and KaTeX, both loaded from a CDN, so you need an internet connection.

Keyboard shortcuts: `→`/`Space` next · `←` previous · `Esc` slide overview · `F` fullscreen · `S` speaker notes.

## Python examples

`examples/` has one short NumPy script per topic (`01_what_is_ml.py` … `08_optimization.py`). Each one reproduces the worked numbers from the slides and then goes a step further, for example training a perceptron on AND, a small network on XOR with backprop, and comparing SGD / Momentum / RMSProp / Adam.

```bash
uv sync                                         # install numpy
uv run python examples/07_backpropagation.py    # run one topic
for f in examples/*.py; do uv run python "$f"; done   # run them all
```

## Exporting to PDF

Open `presentation/ml_fundamentals.html?print-pdf` in Chrome, then use **Print → Save as PDF** with layout set to Landscape and margins set to None.
