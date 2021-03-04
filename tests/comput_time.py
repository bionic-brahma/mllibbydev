from functools import wraps
import time

def track_time(func):
    """ Displaying the run time of the function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f"The function {func.__name__!r} finished in {runtime:.4f} secs")
        return value
    return wrapper


    