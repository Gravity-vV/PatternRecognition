from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Timer:
    elapsed: float = 0.0


@contextmanager
def timed(timer: Timer) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timer.elapsed += time.perf_counter() - start
