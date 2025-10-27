"""Timing utilities for logging with timestamps."""

import time

_start_time = None


def start_timer():
    """Start the global timer."""
    global _start_time
    _start_time = time.time()


def get_elapsed() -> float:
    """Get elapsed time since timer start in seconds."""
    if _start_time is None:
        return 0.0
    return time.time() - _start_time


def log(message: str):
    """Print a message with elapsed time prefix."""
    elapsed = get_elapsed()
    print(f"{elapsed:>6.2f} {message}")
