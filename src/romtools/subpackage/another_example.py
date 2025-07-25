"""This module demonstrates how the API reference page is automatically built."""
import matplotlib.pyplot as plt
import numpy as np


def plot_cosine(x: np.ndarray):
    """Plot a cosine function

    :param x: the independent variable
    """
    plt.plot(x, np.cos(x), '-k')
