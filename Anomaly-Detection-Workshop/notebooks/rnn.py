import numpy as np

# Simulate a 1D RNN with recurrent weight
W = 1  # try <1.0 for vanishing, >1.0 for exploding
h = 0.5  # initial hidden state
T = 20   # number of time steps

grads = []
grad = 1.0  # assume dL/dh_T = 1 for simplicity

for t in range(T):
    grad *= W
    grads.append(grad)

import matplotlib.pyplot as plt
plt.plot(range(T), grads)
plt.title("Gradient Magnitude over Time Steps")
plt.xlabel("Time step (backwards)")
plt.ylabel("Gradient Magnitude")
plt.yscale("log")
plt.grid(True)
plt.show()
