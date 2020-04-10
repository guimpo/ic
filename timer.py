import random
import time


def timerfunc(func):
    """
    A timer decorator
    """
    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "Tempo de execução {func} em segundos: {time}"
        print(msg.format(func=func.__name__,
                         time=runtime))
        return value
    return function_timer
