"""Verify that retries leave the model alone until a timed-out worker exits."""

import unittest
from unittest.mock import MagicMock

from acestep.inference import GenerationConfig, GenerationParams, generate_music


class GenerationRetryTests(unittest.TestCase):
    """Check the inference entrypoint before any model work starts."""

    def test_busy_worker_blocks_retry_before_lm_inference(self) -> None:
        """A timed-out worker's retry should return the busy message immediately."""
        worker = MagicMock()
        worker.is_alive.return_value = True
        dit_handler = MagicMock(_timed_out_generation_workers=(worker,))
        llm_handler = MagicMock(llm_initialized=True)

        result = generate_music(
            dit_handler, llm_handler, GenerationParams(thinking=True), GenerationConfig(),
        )

        self.assertFalse(result.success)
        self.assertIn("timed-out generation is still running", result.error)
        self.assertEqual(dit_handler.mock_calls, [])
        self.assertEqual(llm_handler.mock_calls, [])
