"""Test post-processing of source-preserving edits through the inference API."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from acestep.inference import GenerationConfig, GenerationParams, generate_music


class RepaintPostProcessingTests(unittest.TestCase):
    """Keep the restored waveform intact through inference post-processing."""

    def test_whole_track_processing_only_runs_without_preserved_audio(self) -> None:
        """Source-preserving edits bypass normalization and fades on final export."""
        waveform = torch.full((2, 480), 0.25)
        for preserve_source in (True, False):
            with self.subTest(preserve_source=preserve_source):
                handler = MagicMock(lora_loaded=False, use_lora=False, lora_scale=1.0)
                handler.prepare_seeds.return_value = [17], "17"
                handler.generate_music.return_value = {
                    "success": True,
                    "audios": [{"tensor": waveform.clone(), "sample_rate": 48000}],
                    "repaint_preserved_source": preserve_source,
                }
                params = GenerationParams(
                    task_type="repaint",
                    src_audio="source.wav",
                    repainting_start=0.003,
                    repainting_end=0.006,
                    fade_in_duration=0.001,
                    fade_out_duration=0.001,
                )
                with patch("acestep.inference._load_cached_repaint_source", return_value=None):
                    result = generate_music(handler, None, params, GenerationConfig(batch_size=1))
                self.assertTrue(result.success, result.error)
                output = result.audios[0]["tensor"]
                if preserve_source:
                    torch.testing.assert_close(output, waveform, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(output[:, 0], torch.zeros(2))
                    self.assertGreater(output[0, 240].item(), waveform[0, 240].item())
