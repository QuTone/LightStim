"""Progress reporting helpers for simulation pipeline runs."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


@dataclass
class ProgressSnapshot:
    """Current simulation counters used for progress reporting."""

    shots_total: int
    shots_kept: int
    errors_total: int
    elapsed_sec: float
    max_shots: int
    max_errors: int

    @property
    def ler(self) -> float:
        if self.shots_kept == 0:
            return 0.0
        return self.errors_total / self.shots_kept

    @property
    def eta_sec(self) -> Optional[float]:
        """Estimate time-to-finish using both stop conditions."""
        if self.elapsed_sec <= 0:
            return None
        if self.shots_total >= self.max_shots or self.errors_total >= self.max_errors:
            return 0.0

        eta_candidates = []
        shots_left = max(0, self.max_shots - self.shots_total)
        errors_left = max(0, self.max_errors - self.errors_total)

        shots_rate = self.shots_total / self.elapsed_sec if self.shots_total > 0 else 0.0
        if shots_left > 0 and shots_rate > 0:
            eta_candidates.append(shots_left / shots_rate)

        errors_rate = self.errors_total / self.elapsed_sec if self.errors_total > 0 else 0.0
        if errors_left > 0 and errors_rate > 0:
            eta_candidates.append(errors_left / errors_rate)

        if not eta_candidates:
            return None
        return min(eta_candidates)


class ProgressReporter:
    """Unified reporter with throttling for print/logging outputs."""

    def __init__(
        self,
        *,
        enabled: bool,
        interval_sec: float,
        min_delta_shots: int,
        output: str,
        logger_name: str,
        file_path: Optional[str],
        file_max_bytes: int,
        file_backup_count: int,
        min_emit_gap_sec: float = 1.0,
    ) -> None:
        self.enabled = enabled
        self.interval_sec = max(0.1, interval_sec)
        self.min_delta_shots = max(1, min_delta_shots)
        self.output = output
        self.min_emit_gap_sec = max(0.1, min_emit_gap_sec)

        self._logger = None
        if output in ("logging", "both"):
            self._logger = get_progress_logger(
                logger_name=logger_name,
                file_path=file_path,
                file_max_bytes=file_max_bytes,
                file_backup_count=file_backup_count,
            )

        self._last_emit_monotonic = 0.0
        self._last_emit_shots = 0
        self._has_emitted = False

    def emit(self, snapshot: ProgressSnapshot, *, final: bool = False) -> None:
        if not self.enabled:
            return

        now = time.perf_counter()
        if not final and not self._should_emit(now=now, shots=snapshot.shots_total):
            return

        line = self._format_line(snapshot=snapshot, final=final)
        if self.output in ("print", "both"):
            if final:
                print(f"\r{line}", flush=True)
            else:
                print(f"\r{line}", end="", flush=True)
        if self.output in ("logging", "both") and self._logger is not None:
            self._logger.info(line)

        self._last_emit_monotonic = now
        self._last_emit_shots = snapshot.shots_total
        self._has_emitted = True

    def _should_emit(self, *, now: float, shots: int) -> bool:
        if not self._has_emitted:
            return True

        since_last = now - self._last_emit_monotonic
        if since_last < self.min_emit_gap_sec:
            return False

        if since_last >= self.interval_sec:
            return True

        return (shots - self._last_emit_shots) >= self.min_delta_shots

    @staticmethod
    def _format_line(*, snapshot: ProgressSnapshot, final: bool) -> str:
        prefix = "final " if final else ""
        eta = _format_eta(snapshot.eta_sec)
        return (
            f"{prefix}shots={snapshot.shots_total:,} "
            f"kept={snapshot.shots_kept:,} "
            f"errors={snapshot.errors_total:,} "
            f"LER={snapshot.ler:.2e} "
            f"elapsed={snapshot.elapsed_sec:.1f}s "
            f"ETA={eta}"
        )


def _format_eta(eta_sec: Optional[float]) -> str:
    if eta_sec is None or not math.isfinite(eta_sec):
        return "--"
    if eta_sec < 60:
        return f"{eta_sec:.0f}s"
    minutes = int(eta_sec // 60)
    seconds = int(eta_sec % 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours = int(minutes // 60)
    minutes = int(minutes % 60)
    return f"{hours}h{minutes:02d}m"


def get_progress_logger(
    *,
    logger_name: str,
    file_path: Optional[str],
    file_max_bytes: int,
    file_backup_count: int,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if file_path:
        abs_file = str(Path(file_path).expanduser().resolve())
        file_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, RotatingFileHandler)
            and getattr(h, "baseFilename", None) == abs_file
        ]
        if not file_handlers:
            Path(abs_file).parent.mkdir(parents=True, exist_ok=True)
            rotating = RotatingFileHandler(
                filename=abs_file,
                maxBytes=max(1024, file_max_bytes),
                backupCount=max(1, file_backup_count),
            )
            rotating.setFormatter(formatter)
            logger.addHandler(rotating)

    return logger
