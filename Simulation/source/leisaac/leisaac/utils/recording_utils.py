# -*- coding: utf-8 -*-
"""
leisaac/utils/recording_utils.py

Recorder control + counting helpers for LeIsaac/IsaacLab teleop scripts.

Goals:
- Hide recorder API differences across versions
- Provide consistent resume-count and successful-episode count aggregation
- Provide "num_demos" exit condition helper
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


# -----------------------------
# Low-level: toggle recorder manager
# -----------------------------
def set_recording_enabled(env, enabled: bool) -> bool:
    """
    Best-effort switch for recorder manager across versions.
    Returns True if a compatible API was found and invoked; otherwise False.
    """
    rm = getattr(env, "recorder_manager", None)
    if rm is None:
        return False

    # Preferred setters
    for m in ["set_recording_enabled", "set_enabled"]:
        if hasattr(rm, m):
            try:
                getattr(rm, m)(enabled)
                return True
            except Exception:
                pass

    # Fallback start/stop-ish methods
    for m in (["resume", "start"] if enabled else ["pause", "stop"]):
        if hasattr(rm, m):
            try:
                getattr(rm, m)()
                return True
            except Exception:
                pass

    # Last resort: set boolean attribute
    for a in ["is_recording", "enabled", "recording_enabled"]:
        if hasattr(rm, a):
            try:
                setattr(rm, a, enabled)
                return True
            except Exception:
                pass

    return False


def start_recording(env) -> bool:
    ok = set_recording_enabled(env, True)
    print(f"[record] start -> {'OK' if ok else 'NO-OP (API not found)'}")
    return ok


def stop_recording(env) -> bool:
    ok = set_recording_enabled(env, False)
    print(f"[record] stop -> {'OK' if ok else 'NO-OP (API not found)'}")
    return ok


# -----------------------------
# Counting helpers
# -----------------------------
def get_resume_demo_count(env) -> int:
    """
    Reads existing episode count from underlying dataset file handler (if present).
    Returns 0 if handler is not available.
    """
    rm = getattr(env, "recorder_manager", None)
    if rm is None:
        return 0

    # Most common path in your previous code:
    # env.recorder_manager._dataset_file_handler.get_num_episodes()
    handler = getattr(rm, "_dataset_file_handler", None)
    if handler is not None and hasattr(handler, "get_num_episodes"):
        try:
            return int(handler.get_num_episodes())
        except Exception:
            return 0

    # Alternative layouts
    for attr in ["dataset_file_handler", "_handler", "handler"]:
        h = getattr(rm, attr, None)
        if h is not None and hasattr(h, "get_num_episodes"):
            try:
                return int(h.get_num_episodes())
            except Exception:
                return 0

    return 0


def get_exported_success_count(env) -> int:
    """
    Reads exported_successful_episode_count from recorder_manager if present.
    Returns 0 if missing.
    """
    rm = getattr(env, "recorder_manager", None)
    if rm is None:
        return 0

    for a in ["exported_successful_episode_count", "successful_episode_count", "exported_episode_count"]:
        if hasattr(rm, a):
            try:
                return int(getattr(rm, a))
            except Exception:
                return 0
    return 0


def total_success_demos(env, resume_count: int) -> int:
    return int(resume_count) + get_exported_success_count(env)


@dataclass
class DemoCounter:
    """
    Tracks the last printed total-success count and applies num_demos exit condition.

    Usage:
      counter = DemoCounter(resume_count=..., num_demos=args_cli.num_demos)
      counter.maybe_print_update(env)
      should_exit = counter.reached_limit(env)
    """
    resume_count: int = 0
    num_demos: int = 0
    last_printed_total: int = 0

    def init_from_env(self, env):
        # Initialize last_printed_total as current total
        self.last_printed_total = total_success_demos(env, self.resume_count)

    def maybe_print_update(self, env) -> bool:
        """
        Prints when total_success_demos increases.
        Returns True if printed.
        """
        cur = total_success_demos(env, self.resume_count)
        if cur > self.last_printed_total:
            self.last_printed_total = cur
            print(f"Recorded {cur} successful demonstrations.")
            return True
        return False

    def reached_limit(self, env) -> bool:
        """
        Returns True when num_demos > 0 and total_success_demos >= num_demos.
        """
        if int(self.num_demos) <= 0:
            return False
        cur = total_success_demos(env, self.resume_count)
        return cur >= int(self.num_demos)
