# -*- coding: utf-8 -*-
"""
leisaac/utils/rate_limiter.py

Utility for enforcing a fixed stepping rate while allowing rendering.
"""

import time


class RateLimiter:
    """Convenience class for enforcing a fixed loop rate.

    Typical usage:
        rate = RateLimiter(hz=60)
        while running:
            ...
            rate.sleep(env)
    """

    def __init__(self, hz: int):
        if hz <= 0:
            raise ValueError(f"hz must be positive, got {hz}")
        self.hz = float(hz)
        self.sleep_duration = 1.0 / self.hz
        self.render_period = min(0.0166, self.sleep_duration)
        self.last_time = time.time()

    def reset(self):
        """Reset the internal timer (useful after env.reset())."""
        self.last_time = time.time()

    def sleep(self, env):
        """Sleep until the next tick while rendering the simulator."""
        next_wakeup_time = self.last_time + self.sleep_duration

        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            # keep simulator responsive
            if hasattr(env, "sim"):
                env.sim.render()

        self.last_time += self.sleep_duration

        # catch up if loop was too slow
        now = time.time()
        if self.last_time < now:
            while self.last_time < now:
                self.last_time += self.sleep_duration
