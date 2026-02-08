import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Define the range [-10, 10]
x = np.linspace(-10, 10, 400)
y = sigmoid_derivative(x)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Sigmoid Derivative", color='teal', linewidth=2)
plt.title("Derivative of the Sigmoid Activation Function")
plt.xlabel("Input (x)")
plt.ylabel("Gradient")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()