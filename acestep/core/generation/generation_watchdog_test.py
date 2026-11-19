"""Regression tests for generation deadlines and retries after a timeout."""

import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from acestep.core.generation import generation_watchdog


class GenerationWatchdogTests(unittest.TestCase):
    """Verify long progressing runs and genuinely stalled workers."""

    def setUp(self) -> None:
        """Run against a manually advanced clock and an isolated handler."""
        self.seconds = 0
        self.handler = SimpleNamespace()
        self.report_step = MagicMock()
        self.enterContext(patch.object(generation_watchdog, "monotonic", lambda: self.seconds))
        self.enterContext(patch.dict(os.environ))
        os.environ.pop("ACESTEP_GENERATION_TIMEOUT", None)

    def test_32_step_generation_can_take_over_600_seconds(self) -> None:
        """Each completed step extends the default deadline during a slow run."""
        def generate(step_completed):
            """Replay a slow 32-step sampler without GPU work or sleeps."""
            for step in range(1, 33):
                self.seconds += 44
                step_completed(step, 32)
            return {"target_latents": "finished"}

        output = generation_watchdog.run_generation_worker(
            self.handler, generate, self.report_step, "batch=2, duration=198s",
        )
        self.assertEqual(self.seconds, 1408)
        self.assertEqual(output["target_latents"], "finished")
        self.assertFalse(generation_watchdog.timed_out_generation_running(self.handler))

    def test_late_step_after_inactivity_is_cancelled(self) -> None:
        """Report the most recent step when a later step takes over 600 seconds."""
        def generate(step_completed):
            """Complete one step, then exceed the inactivity limit."""
            self.seconds = 44
            step_completed(1, 32)
            self.seconds = 645
            step_completed(2, 32)
            self.fail("The sampler continued past its timeout")

        with self.assertRaisesRegex(
            TimeoutError, r"600 seconds without a completed diffusion step.*1/32",
        ):
            generation_watchdog.run_generation_worker(
                self.handler, generate, self.report_step, "batch=2, duration=198s",
            )

    def test_explicit_timeout_remains_a_wall_clock_limit(self) -> None:
        """Step updates do not extend an explicit ACESTEP_GENERATION_TIMEOUT."""
        os.environ["ACESTEP_GENERATION_TIMEOUT"] = "600"

        def generate(step_completed):
            """Report steady progress across the configured wall-clock limit."""
            for step in range(1, 33):
                self.seconds += 44
                step_completed(step, 32)
            self.fail("The sampler continued past its wall-clock timeout")

        with self.assertRaisesRegex(TimeoutError, r"timed out after 600 seconds\. .*13/32"):
            generation_watchdog.run_generation_worker(
                self.handler, generate, self.report_step, "explicit timeout",
            )

    def test_worker_error_is_returned_to_caller(self) -> None:
        """A real model error should surface before the deadline expires."""
        generate = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            generation_watchdog.run_generation_worker(
                self.handler, generate, self.report_step, "failed generation",
            )
        self.assertFalse(generation_watchdog.timed_out_generation_running(self.handler))

    def test_timeout_blocks_retry_until_worker_leaves(self) -> None:
        """A timed-out blocking operation keeps ownership of the handler until it exits."""
        finished_step = threading.Event()
        release = threading.Event()

        def generate(step_completed):
            """Wait inside an operation, then try to resume diffusion."""
            step_completed(1, 32)
            release.wait(5)
            step_completed(2, 32)
            finished_step.set()
            return {}

        def report(_step, _total):
            """Age the sampler's last completed step while it is still blocked."""
            self.seconds = 601

        try:
            with self.assertRaisesRegex(TimeoutError, "600 seconds without"):
                generation_watchdog.run_generation_worker(
                    self.handler, generate, report, "blocked generation",
                )
            retry = MagicMock(return_value={"target_latents": "retried"})
            with self.assertRaisesRegex(RuntimeError, "timed-out generation is still running"):
                generation_watchdog.run_generation_worker(self.handler, retry, None, "retry")
            retry.assert_not_called()
        finally:
            release.set()
            for worker in getattr(self.handler, "_timed_out_generation_workers", ()):
                worker.join(timeout=2)

        self.assertFalse(finished_step.is_set())
        self.assertFalse(generation_watchdog.timed_out_generation_running(self.handler))
        output = generation_watchdog.run_generation_worker(self.handler, retry, None, "retry")
        self.assertEqual(output, {"target_latents": "retried"})
