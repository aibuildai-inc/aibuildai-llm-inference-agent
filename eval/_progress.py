"""Shared progress bar for the evaluation arms.

Written to /dev/tty, NOT to stdout or stderr, because run_eval.sh pipes each arm
through `tail -6`: a pipe buffers the whole stream, so anything printed normally
appears only after the arm has already finished. Writing to the controlling
terminal bypasses the pipe and shows progress as it happens. With no terminal
(CI, nohup) it degrades to one line per update on stderr, and if even that is
unavailable it does nothing at all -- a progress bar must never be able to fail
a measurement.
"""
from __future__ import annotations

import os
import sys
import time


class Progress:
    def __init__(self, total: int, label: str = "", width: int = 28):
        self.total = max(int(total), 1)
        self.label = label
        self.width = width
        self.n = 0
        self.t0 = time.time()
        self._tty = None
        self._plain = False
        try:
            self._tty = open("/dev/tty", "w")
        except OSError:
            self._plain = True

    def _emit(self, text: str, end: str) -> None:
        try:
            if self._tty is not None:
                self._tty.write(text + end)
                self._tty.flush()
            elif self._plain:
                sys.stderr.write(text + ("\n" if end == "\r" else end))
                sys.stderr.flush()
        except Exception:
            self._tty, self._plain = None, False   # never break the run over a bar

    def update(self, n: int = 1, note: str = "") -> None:
        self.n += n
        frac = min(self.n / self.total, 1.0)
        filled = int(frac * self.width)
        bar = "#" * filled + "-" * (self.width - filled)
        el = time.time() - self.t0
        eta = (el / frac - el) if frac > 0 else 0.0
        self._emit(
            f"\r  {self.label} [{bar}] {self.n}/{self.total}"
            f"  {el/60:5.1f}m elapsed  {eta/60:5.1f}m left{('  ' + note) if note else ''}   ",
            "\r" if self._plain else "",
        )

    def close(self, note: str = "") -> None:
        el = time.time() - self.t0
        self._emit(f"\r  {self.label} [{'#' * self.width}] {self.n}/{self.total}"
                   f"  done in {el/60:.1f}m{('  ' + note) if note else ''}   ", "\n")
        if self._tty is not None:
            try: self._tty.close()
            except Exception: pass
            self._tty = None
