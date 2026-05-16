# 🧠 Neural Network from Scratch

A fully functional 2-layer neural network built using **only NumPy and Pandas** — no PyTorch, no TensorFlow, no shortcuts. Trained on the MNIST handwritten digit dataset.

---

## How It Works

The network learns to recognize handwritten digits (0–9) by doing the math manually:

1. Each 28×28 image is flattened into 784 pixel values
2. Pixel values are normalized to the range [0, 1]
3. Data flows forward through two layers
4. Errors flow backward and update weights (backpropagation)
5. Repeat 1000 times until the network gets good at guessing

---

## Architecture

```
Input Layer          Hidden Layer         Output Layer
  (784)      →          (10)       →          (10)

 [pixel 1]         [neuron 1]           [digit 0]
 [pixel 2]         [neuron 2]           [digit 1]
    ...        →      ...        →          ...
    ...                ...               [digit 8]
 [pixel 784]       [neuron 10]          [digit 9]

              W1 (10×784)        W2 (10×10)
              + B1 (10×1)        + B2 (10×1)
              → ReLU             → Softmax

                 ↕ Backpropagation updates W1,B1,W2,B2
```

**Activation Functions:**
- Hidden Layer → `ReLU(Z) = max(0, Z)` — kills negatives, keeps positives
- Output Layer → `Softmax(Z) = e^Z / Σe^Z` — converts scores to probabilities

---

## Project Structure

```
neural-network-from-scratch/
│
├── Neural_Network_from_Scratch.ipynb   # Main notebook
└── README.md
```

---

## Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/your-username/neural-network-from-scratch.git
cd neural-network-from-scratch
```

**2. Install dependencies**
```bash
pip install numpy pandas matplotlib kagglehub
```

**3. Open and run the notebook**
```bash
jupyter notebook Neural_Network_from_Scratch.ipynb
```

The notebook downloads the MNIST dataset automatically via `kagglehub`.

---

## What's Inside the Code

| Function | What It Does |
|---|---|
| `initialize_parameters()` | Creates random weights W1, B1, W2, B2 |
| `forward_propagation()` | Passes input through the two layers |
| `ReLU()` | Activation for hidden layer |
| `softmax()` | Converts output to probabilities |
| `one_hot_converter()` | Encodes labels (e.g. 3 → [0,0,0,1,0,...]) |
| `back_propagation()` | Computes gradients using chain rule |
| `update_parameters()` | Adjusts weights using learning rate |
| `gradient_descent()` | Runs the full training loop |
| `get_accuracy()` | Checks how many predictions are correct |

---

## Training Details

| Setting | Value |
|---|---|
| Dataset | MNIST (60,000 images) |
| Train / Val split | 80% / 20% |
| Input size | 784 (28×28 pixels) |
| Hidden neurons | 10 |
| Output classes | 10 (digits 0–9) |
| Learning rate | 0.10 |
| Iterations | 1000 |

---

## The Math (Simplified)

**Forward Pass:**
```
Z1 = W1 · X + B1       ← linear transform
A1 = ReLU(Z1)          ← activation
Z2 = W2 · A1 + B2      ← linear transform
A2 = Softmax(Z2)       ← probabilities
```

**Backward Pass (Gradient Descent):**
```
dZ2 = A2 - Y_one_hot
dW2 = (1/m) * dZ2 · A1ᵀ
dZ1 = W2ᵀ · dZ2 * ReLU'(Z1)
dW1 = (1/m) * dZ1 · Xᵀ
```

**Weight Update:**
```
W = W - learning_rate * dW
B = B - learning_rate * dB
```

---

## Stack

- **Python**
- **NumPy** — all matrix math
- **Pandas** — loading the CSV dataset
- **Matplotlib** — visualizing predictions

---

## Why Build This?

Most people use frameworks like PyTorch or TensorFlow without understanding what's happening underneath. This project builds everything from scratch to show how the math actually works — forward pass, backpropagation, gradient descent, all by hand.

---

## License

MIT — use it, learn from it, break it.
