# Artur Szcześniak
# http://github.com/feveria
#
# Simple neural network learning to plot sin(x)


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import statistics
import torch.nn as nn
import torch

# Constants
CYCLES = 2
LEN = 2 * math.pi * CYCLES
LR = 0.1
X_RESOLUTION = 0.005
ERR_FREQ = 100
INTERVAL = 16

# Activation function
def nonlin(x):
    fun = nn.LeakyReLU(0.1)
    tensor = torch.Tensor(x)
    postx = fun(tensor)
    return np.array(postx)


def main():

    # Random seed, afair good practice
    np.random.seed(12101990)

    # Tensor initalization
    x = np.arange(0, LEN, X_RESOLUTION)
    syn0 = (np.random.random(np.shape(x)))
    y = (np.sin(x) + 2) / 2

    def epoch(i, syn0):
        # Forward propagation
        l0 = np.ones(len(x)) / 2
        l1 = nonlin(l0 * syn0)

        # Calculate the error
        l1_error = y - l1

        # Multiply error by the nonlin function
        l1_delta = l1_error * nonlin(l1)

        # Update weights
        syn0 += l0 * l1_delta * LR

        # Print mean error every 100 cycles
        if i % ERR_FREQ == 0:
            print("Mean error: {}".format(abs(statistics.mean(l1_error))))

        # Return data for animation class
        line.set_data(x, 2*(l0 * syn0) - 2)
        return line,

    # Animiation initialization function
    def init():
        line.set_data([], [])
        return line,

    # Animation figure setup
    fig = plt.figure()
    fig.set_size_inches(16, 9)
    ax = plt.axes(xlim=(0, LEN), ylim=(-2, 2))
    line, = ax.plot([], [], lw=2)

    # Execute an epoch every few ms and animate
    anim = animation.FuncAnimation(fig, epoch, init_func=init, fargs=(syn0,), interval=INTERVAL, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
