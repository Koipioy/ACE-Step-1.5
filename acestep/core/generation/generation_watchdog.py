"""Monitor diffusion progress and the lifetime of the generation worker."""

import os
import threading
from time import monotonic
from typing import Any, Callable

from loguru import logger

_DEFAULT_TIMEOUT_SECONDS = 600
_worker_lock = threading.Lock()
GENERATION_BUSY_MESSAGE = (
    "A timed-out generation is still running on this model. Wait for it to finish "
    "before retrying. If it remains stuck, restart ACE-Step."
)

StepCallback = Callable[[int, int], None]


class _GenerationWatchdog:
    """Wait for a worker to finish, with an optional deadline renewed by completed steps."""

    def __init__(self, timeout: int, renew_on_step: bool) -> None:
        """Start the timeout clock and initialize thread-safe progress state."""
        self._condition = threading.Condition()
        self._timeout = timeout
        self._renew_on_step = renew_on_step
        self._deadline = monotonic() + timeout
        self._finished = False
        self._timed_out = False
        self.last_step: tuple[int, int] | None = None

    def step_completed(self, step: int, total: int) -> None:
        """Record completed sampler work and stop at the next step after a timeout."""
        with self._condition:
            now = monotonic()
            if self._timed_out or now >= self._deadline:
                self._timed_out = True
                self._condition.notify()
                raise TimeoutError("Diffusion stopped after the generation timeout.")
            self.last_step = (step, total)
            if self._renew_on_step:
                self._deadline = now + self._timeout
            self._condition.notify()

    def finish(self) -> None:
        """Wake the caller once the worker has returned or raised."""
        with self._condition:
            self._finished = True
            if monotonic() >= self._deadline:
                self._timed_out = True
            self._condition.notify()

    def wait(self, report_step: StepCallback | None) -> bool:
        """Report new steps on the calling thread and return whether the worker finished."""
        reported_step = None
        while True:
            with self._condition:
                if self._timed_out:
                    return False
                if self._finished:
                    return True
                remaining = self._deadline - monotonic()
                if remaining <= 0:
                    self._timed_out = True
                    return False
                if self.last_step == reported_step:
                    self._condition.wait(remaining)
                    continue
                reported_step = self.last_step

            if report_step is not None and reported_step is not None:
                try:
                    report_step(*reported_step)
                except Exception as exc:
                    logger.debug("Unable to report diffusion progress: {}", exc)


def timed_out_generation_running(handler: Any) -> bool:
    """Return whether a timed-out worker can still use this handler's model."""
    workers = getattr(handler, "_timed_out_generation_workers", ())
    return any(worker.is_alive() for worker in workers)


def run_generation_worker(
    handler: Any,
    generate: Callable[[StepCallback | None], dict[str, Any]],
    report_step: StepCallback | None,
    request_description: str,
) -> dict[str, Any]:
    """Run generation and preserve the worker guard until its thread actually exits."""
    configured_timeout = os.environ.get("ACESTEP_GENERATION_TIMEOUT")
    timeout = int(configured_timeout) if configured_timeout is not None else _DEFAULT_TIMEOUT_SECONDS
    renew_on_step = report_step is not None and configured_timeout is None
    watchdog = _GenerationWatchdog(timeout, renew_on_step)
    result: dict[str, Any] = {}
    error: Exception | None = None

    def run() -> None:
        """Transfer the worker's result or exception to the waiting caller."""
        nonlocal result, error
        try:
            result = generate(watchdog.step_completed if report_step is not None else None)
        except Exception as exc:
            error = exc
        finally:
            watchdog.finish()

    worker = threading.Thread(target=run, name="service-generate", daemon=True)
    with _worker_lock:
        if timed_out_generation_running(handler):
            raise RuntimeError(GENERATION_BUSY_MESSAGE)
        worker.start()

    if not watchdog.wait(report_step):
        message = f"Music generation timed out after {timeout} seconds"
        if renew_on_step:
            message += " without a completed diffusion step"
        message += "."
        if watchdog.last_step is not None:
            step, total = watchdog.last_step
            message += f" Last completed diffusion step: {step}/{total}."
        with _worker_lock:
            if worker.is_alive():
                previous = getattr(handler, "_timed_out_generation_workers", ())
                active = tuple(thread for thread in previous if thread.is_alive())
                handler._timed_out_generation_workers = active + (worker,)
                message += f" {GENERATION_BUSY_MESSAGE}"
        logger.error("[generate_music] {} ({})", message, request_description)
        raise TimeoutError(message)

    worker.join()
    if error is not None:
        raise error
    return result
