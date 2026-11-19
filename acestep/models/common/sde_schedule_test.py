"""Regression tests for the noise schedule used by Base and SFT sampling."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from acestep.models.base import modeling_acestep_v15_base as base
from acestep.models.sft import modeling_acestep_v15_base as sft
from acestep.models.xl_base import modeling_acestep_v15_xl_base as xl_base
from acestep.models.xl_sft import modeling_acestep_v15_xl_base as xl_sft


class SdeScheduleTests(unittest.TestCase):
    """Check that each model evaluation receives latents at its declared timestep."""

    def _assert_schedule(
        self, schedule: list[float], variants: tuple = (base, sft, xl_base, xl_sft), **sampler_kwargs
    ) -> None:
        """Run the real sampler with a deterministic decoder for straight-line flow."""
        clean = torch.tensor([[[-0.5], [0.25]], [[0.5], [-0.25]]])
        noise = torch.ones_like(clean)
        expected_timesteps = torch.tensor(schedule[:-1])
        expected_latents = torch.lerp(clean, noise, expected_timesteps[:, None, None, None])

        for variant in variants:
            with self.subTest(variant=variant.__name__):
                model = variant.AceStepConditionGenerationModel
                host = SimpleNamespace(
                    decoder=MagicMock(return_value=(noise - clean, None)),
                    prepare_condition=MagicMock(return_value=(None, None, clean)),
                    prepare_noise=MagicMock(return_value=noise),
                )
                host.get_x0_from_noise = model.get_x0_from_noise.__get__(host)
                host.renoise = model.renoise.__get__(host)
                with patch("torch.randn_like", side_effect=torch.ones_like):
                    result = model.generate_audio(
                        host,
                        text_hidden_states=None,
                        text_attention_mask=None,
                        lyric_hidden_states=None,
                        lyric_attention_mask=None,
                        refer_audio_acoustic_hidden_states_packed=None,
                        refer_audio_order_mask=None,
                        src_latents=clean,
                        chunk_masks=None,
                        is_covers=None,
                        diffusion_guidance_scale=1.0,
                        dcw_enabled=False,
                        use_progress_bar=False,
                        **sampler_kwargs,
                    )

                calls = host.decoder.call_args_list
                actual_timesteps = torch.stack([call.kwargs["timestep"] for call in calls])
                torch.testing.assert_close(
                    actual_timesteps, expected_timesteps[:, None].expand(-1, clean.shape[0])
                )
                actual_latents = torch.stack([call.kwargs["hidden_states"] for call in calls])
                torch.testing.assert_close(actual_latents, expected_latents)
                torch.testing.assert_close(result["target_latents"], clean)

    def test_uniform_sde_schedule(self) -> None:
        """SDE follows the default linear schedule through the clean endpoint."""
        self._assert_schedule([1.0, 0.75, 0.5, 0.25, 0.0], infer_method="sde", infer_steps=4)

    def test_shifted_sde_schedule(self) -> None:
        """SDE re-noises to the shifted timestep passed to the next decoder evaluation."""
        self._assert_schedule(
            [1.0, 0.9, 0.75, 0.5, 0.0], infer_method="sde", infer_steps=4, shift=3.0
        )

    def test_custom_sde_schedule(self) -> None:
        """Explicit SDE timesteps override the configured step count and shift."""
        schedule = [1.0, 0.8, 0.45, 0.1, 0.0]
        self._assert_schedule(
            schedule, infer_method="sde", infer_steps=8, shift=3.0, timesteps=torch.tensor(schedule)
        )

    def test_truncated_sde_schedule_for_covers(self) -> None:
        """Cover initialization preserves the remaining noise levels after truncation."""
        self._assert_schedule(
            [0.75, 9 / 14, 0.5, 0.3, 0.0],
            infer_method="sde",
            infer_steps=8,
            shift=3.0,
            cover_noise_strength=0.25,
        )

    def test_shifted_ode_schedule(self) -> None:
        """Deterministic Euler sampling still follows the shifted schedule."""
        self._assert_schedule(
            [1.0, 0.9, 0.75, 0.5, 0.0], infer_method="ode", infer_steps=4, shift=3.0
        )

    def test_xl_base_reports_the_effective_step_count(self) -> None:
        """Progress reports completed steps from the actual custom schedule."""
        schedule = [1.0, 0.75, 0.4, 0.0]
        callback = MagicMock()
        self._assert_schedule(
            schedule, variants=(xl_base,), infer_steps=32,
            timesteps=torch.tensor(schedule), on_diffusion_step=callback,
        )
        self.assertEqual(callback.call_args_list, [call(1, 3), call(2, 3), call(3, 3)])
