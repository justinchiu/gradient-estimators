import numpy as onp

import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = onp.get_printoptions()
    onp.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        onp.set_printoptions(**original)

