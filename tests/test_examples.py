import numpy as np

import romtools
from romtools.example import my_example
from romtools.subpackage.another_example import plot_cosine


def test_all_examples():
    print(f'romtools version: { romtools.__version__ }')
    my_example()
    plot_cosine(np.linspace(0, 1, 100))
    print(f'Test passed!')
