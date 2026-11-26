"""Optional sampling of Clifford+T circuit text with Clifft."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import RLock, Thread
from typing import TextIO

import numpy as np


_TAG = re.compile(r"(?m)^([ \t]*[A-Za-z_][A-Za-z_0-9]*)\[[^\]\r\n]*\]")
_SEED_RANGE = 1 << 64


@dataclass(frozen=True)
class ClifftBatch:
    """Shot-major measurement bits, raw record parities, and EXP_VAL values."""

    detectors: np.ndarray
    observables: np.ndarray
    measurements: np.ndarray
    expectations: np.ndarray


class ClifftSampler:
    """Compile a physical circuit once in a separate Clifft Python process.

    Use as a context manager or call ``close()`` when finished. Detector and
    observable columns are raw XORs of the declared records. Each call uses
    ``seed + batch_index`` modulo 2**64, starting with batch index zero.
    """

    def __init__(
        self,
        circuit_text: str,
        *,
        python_executable: str | os.PathLike[str] | None = None,
        seed: int = 0,
        threads: int = 1,
        max_peak_rank: int = 24,
        timeout: float = 300.0,
    ) -> None:
        if not isinstance(circuit_text, str):
            raise TypeError("circuit_text must be Clifford+T circuit text")
        for name, value, minimum in (
            ("seed", seed, 0), ("threads", threads, 1), ("max_peak_rank", max_peak_rank, 0)
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if seed >= _SEED_RANGE:
            raise ValueError("seed must be less than 2**64")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")

        executable = python_executable
        if executable is None:
            executable = os.environ.get("CLIFFT_PYTHON") or sys.executable
        self._seed = seed
        self._batch_index = 0
        self._timeout = timeout
        self._lock = RLock()
        self._process: subprocess.Popen[str] | None = None
        self._reader: Thread | None = None
        self._responses: Queue[str | None] = Queue()
        self._directory = TemporaryDirectory(prefix="lightstim-clifft-")
        directory = Path(self._directory.name)
        self._output_path = directory / "batch.npz"
        self._stderr_path = directory / "stderr.log"
        config_path = directory / "config.json"
        try:
            config_path.write_text(json.dumps({
                "circuit_text": _TAG.sub(r"\1", circuit_text),
                "threads": threads,
                "max_peak_rank": max_peak_rank,
            }), encoding="utf-8")
            worker = Path(__file__).with_name("_clifft_worker.py")
            with self._stderr_path.open("wb") as stderr:
                try:
                    # Keep the interpreter's venv path, including any symlink.
                    self._process = subprocess.Popen(
                        [os.fspath(executable), "-u", str(worker), str(config_path)],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=stderr,
                        encoding="utf-8",
                        errors="replace",
                    )
                except OSError as error:
                    raise RuntimeError(f"Could not start Clifft worker using {executable}: {error}") from error
            self._reader = Thread(target=self._read_responses, args=(self._process.stdout,), daemon=True)
            self._reader.start()
            response = self._receive("compilation", "widths")
            self._widths = response["widths"]
        except BaseException:
            self.close()
            raise

    def _read_responses(self, stdout: TextIO) -> None:
        try:
            for line in stdout:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def _receive(self, operation: str, expected_key: str) -> dict:
        try:
            line = self._responses.get(timeout=self._timeout)
        except Empty as error:
            raise TimeoutError(f"Clifft worker timed out during {operation}") from error
        if line is None:
            with self._stderr_path.open("rb") as stderr:
                stderr.seek(max(0, self._stderr_path.stat().st_size - 4096))
                detail = stderr.read().decode(errors="replace").strip()
            raise RuntimeError(f"Clifft worker exited during {operation}. {detail}".rstrip())
        try:
            response = json.loads(line)
        except ValueError as error:
            raise RuntimeError(f"Clifft worker returned an invalid response during {operation}") from error
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Clifft worker failed during {operation}: {response['error']}")
        if not isinstance(response, dict) or expected_key not in response:
            raise RuntimeError(f"Clifft worker returned an invalid response during {operation}")
        return response

    def sample_batch(self, shots: int) -> ClifftBatch:
        """Sample ``shots`` trajectories, keeping every row in the returned arrays."""
        if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
            raise ValueError("shots must be a positive integer")
        with self._lock:
            if self._process is None:
                raise RuntimeError("Clifft sampler is closed")
            request = {"shots": shots, "seed": (self._seed + self._batch_index) % _SEED_RANGE}
            try:
                try:
                    self._process.stdin.write(json.dumps(request) + "\n")
                    self._process.stdin.flush()
                except BrokenPipeError as error:
                    raise RuntimeError("Clifft worker exited before sampling") from error
                self._receive("sampling", "ok")
                with np.load(self._output_path, allow_pickle=False) as data:
                    arrays = {name: data[name] for name in self._widths}
                for name, width in self._widths.items():
                    if arrays[name].shape != (shots, width):
                        raise RuntimeError(f"Clifft worker returned the wrong shape for {name}")
                self._output_path.unlink()
            except BaseException:
                self.close()
                raise
            self._batch_index += 1
            return ClifftBatch(**arrays)

    def close(self) -> None:
        """Stop the worker and release its temporary files."""
        with self._lock:
            process = self._process
            self._process = None
            if process is not None:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                if self._reader is not None and self._reader.ident is not None:
                    self._reader.join()
                process.stdout.close()
            self._directory.cleanup()

    def __enter__(self) -> ClifftSampler:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
