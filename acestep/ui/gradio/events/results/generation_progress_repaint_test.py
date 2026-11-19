"""Verify the conditioning submitted by Gradio for lyric edits."""

import inspect
import unittest
from unittest.mock import MagicMock, patch

from acestep.constants import TASK_INSTRUCTIONS
from acestep.inference import GenerationParams, GenerationResult
from acestep.ui.gradio.events.results import generation_progress


class RepaintGenerationRequestTests(unittest.TestCase):
    """Ensure hidden Remix state cannot change a Repaint request."""

    def _capture_request(self, task_type: str) -> GenerationParams:
        """Run the Gradio generation entrypoint up to the inference call."""
        generate = generation_progress.generate_with_progress
        inputs = {
            name: None for name, parameter in inspect.signature(generate).parameters.items()
            if parameter.default is inspect.Parameter.empty
        }
        inputs.update(
            task_type=task_type,
            src_audio="source.wav",
            lyrics="[Verse]\nFull verse with the revised line",
            vocal_language="en",
            repainting_start=15.0,
            repainting_end=20.0,
            repaint_mode="balanced",
            repaint_strength=0.4,
            inference_steps=32,
            audio_format="flac",
            instruction_display_gen=TASK_INSTRUCTIONS["cover"],
            audio_cover_strength=0.0,
            cover_noise_strength=0.2,
            progress=MagicMock(),
        )
        result = GenerationResult(success=False, status_message="stopped after request capture")
        with patch.object(generation_progress, "generate_music", return_value=result) as inference:
            events = generate(**inputs)
            next(events)
            events.close()
            inference.assert_called_once()
            return inference.call_args.kwargs["params"]

    def test_repaint_uses_source_context_and_repaint_instruction(self) -> None:
        """A previous Remix cannot switch a timed lyric edit to text-to-music."""
        params = self._capture_request("repaint")

        self.assertEqual(params.instruction, TASK_INSTRUCTIONS["repaint"])
        self.assertEqual(params.audio_cover_strength, 1.0)
        self.assertEqual(params.cover_noise_strength, 0.0)
        self.assertEqual(params.chunk_mask_mode, "explicit")
        self.assertEqual((params.repainting_start, params.repainting_end), (15.0, 20.0))
        self.assertEqual(params.lyrics, "[Verse]\nFull verse with the revised line")
        self.assertEqual(params.vocal_language, "en")
        self.assertEqual(params.repaint_strength, 0.4)

    def test_remix_keeps_its_own_strengths(self) -> None:
        """Remix requests retain the source controls exposed by that mode."""
        params = self._capture_request("cover")

        self.assertEqual(params.instruction, TASK_INSTRUCTIONS["cover"])
        self.assertEqual(params.audio_cover_strength, 0.0)
        self.assertEqual(params.cover_noise_strength, 0.2)
