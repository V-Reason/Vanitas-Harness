"""空闲看门狗：单守护线程，空闲超时后回调一次；pause/resume 供重启与长命令使用。"""
from __future__ import annotations

import sys
import threading
import time
from typing import Callable

from ..ui import Palette


class IdleWatchdog:
    """单守护线程：空闲超时后回调一次；pause/resume 供重启与长命令使用。"""

    def __init__(
        self,
        timeout_sec: float,
        interval_sec: float,
        on_idle: Callable[[], None],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.timeout_sec = max(0.0, float(timeout_sec))
        self.interval_sec = max(0.01, float(interval_sec))
        self._on_idle = on_idle
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_active = clock()
        self._paused = False
        self._fired = False

    def start(self) -> None:
        if self.timeout_sec <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="vanitas-idle-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def touch(self) -> None:
        with self._lock:
            self._last_active = self._clock()
            self._fired = False

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            self._last_active = self._clock()

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._last_active = self._clock()
            self._fired = False

    def remaining(self) -> float | None:
        with self._lock:
            if self.timeout_sec <= 0:
                return None
            if self._fired:
                return 0.0
            return max(0.0, self.timeout_sec - (self._clock() - self._last_active))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            with self._lock:
                if self._paused or self._fired or self.timeout_sec <= 0:
                    continue
                if self._clock() - self._last_active < self.timeout_sec:
                    continue
                self._fired = True
            try:
                self._on_idle()
            except Exception as exc:  # 守护线程不能因回调异常静默退出
                print(
                    Palette({}, enabled=False).err(f"空闲看门狗异常：{exc}"),
                    file=sys.stderr,
                )
