"""An example of snippets to include in documentation directly from source code."""


def my_example(a: int = 0) -> None:
    """An example docstring in sphinx format.

    !!! Note
        Admonitions will be displayed using the triple ! format when rendered via mkdocs. See
        [mkdocs-material](https://squidfunk.github.io/mkdocs-material/reference/admonitions/). You can also show latex
        equations using the normal $y=f(x)$ format. Most other markdown syntax can be used directly in the docstring
        and will render properly when pulled into the documentation using `mkdocstrings`.

    :param a: this is an example sphinx documentation format for a parameter. Use `a: int` type hints to render
              the type of the parameter through `mkdocstrings`.
    :returns: `None`, this function does not have a return value
    """
    # All the lines below will be cut and paste into a doc code block that references the "example_name" code snippet
    # --8<-- [start:example_name]
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 1, 100)
    y = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y, '-k')
    # --8<-- [end:example_name]
