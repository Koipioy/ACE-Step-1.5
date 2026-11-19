"""Waveform-level splice for repaint tasks.

After VAE decode, non-repaint regions still carry VAE reconstruction error.
This module replaces those regions with the original user waveform and applies
a short crossfade at boundaries to eliminate clicks.
"""

from typing import List

import torch
from loguru import logger


def _build_waveform_crossfade_mask(
    total_samples: int,
    start_sample: int,
    end_sample: int,
    crossfade_samples: int,
    device: torch.device,
    *,
    crossfade_inside: bool = False,
) -> torch.Tensor:
    """Build a per-sample repaint mask with optional ramps at the boundaries.

    Args:
        total_samples: Total waveform length in samples.
        start_sample: First sample of the repaint region.
        end_sample: One-past-last sample of the repaint region.
        crossfade_samples: Number of samples over which to ramp.
        device: Target tensor device.
        crossfade_inside: Keep boundary ramps within the selected range.

    Returns:
        Float tensor of shape ``[total_samples]`` with values in ``[0, 1]``.
    """
    mask = torch.zeros(total_samples, device=device)
    mask[start_sample:end_sample] = 1.0

    if crossfade_samples <= 0:
        return mask

    if crossfade_inside:
        ramp_len = min(crossfade_samples, (end_sample - start_sample - 1) // 2)
        if ramp_len > 0:
            ramp = torch.arange(ramp_len, device=device, dtype=mask.dtype) / ramp_len
            if start_sample > 0:
                mask[start_sample:start_sample + ramp_len] = ramp
            if end_sample < total_samples:
                mask[end_sample - ramp_len:end_sample] = ramp.flip(0)
        return mask

    fade_start = max(start_sample - crossfade_samples, 0)
    ramp_len = start_sample - fade_start
    if ramp_len > 0:
        ramp = torch.linspace(0.0, 1.0, ramp_len + 2, device=device)[1:-1]
        mask[fade_start:start_sample] = ramp

    fade_end = min(end_sample + crossfade_samples, total_samples)
    ramp_len = fade_end - end_sample
    if ramp_len > 0:
        ramp = torch.linspace(1.0, 0.0, ramp_len + 2, device=device)[1:-1]
        mask[end_sample:fade_end] = ramp

    return mask


def apply_repaint_waveform_splice(
    pred_wavs: torch.Tensor,
    src_wavs: torch.Tensor,
    repainting_starts: List[float],
    repainting_ends: List[float],
    sample_rate: int = 48000,
    crossfade_duration: float = 0.01,
    *,
    preserve_source: bool = False,
) -> torch.Tensor:
    """Splice original user waveform into non-repaint regions after VAE decode.

    Args:
        pred_wavs: VAE-decoded waveform ``[B, C, samples]``.
        src_wavs: Original user waveform (pre-VAE) ``[B, C, samples]``.
        repainting_starts: Per-batch start time in seconds.
        repainting_ends: Per-batch end time in seconds.
        sample_rate: Audio sample rate (default 48000).
        crossfade_duration: Crossfade length in seconds at splice boundaries
            (default 0.01 = 10 ms).
        preserve_source: Return the padded source length and crossfade inside the edit.

    Returns:
        Spliced waveform with the source length when preserving it, or the decoded length.
    """
    B = pred_wavs.shape[0]
    min_samples = min(pred_wavs.shape[-1], src_wavs.shape[-1])
    splice_samples = src_wavs.shape[-1] if preserve_source else min_samples

    if pred_wavs.shape[-1] != src_wavs.shape[-1]:
        logger.debug(
            "[repaint_waveform_splice] Length mismatch: pred_wavs={}, src_wavs={}; "
            "splicing {} shared samples",
            pred_wavs.shape[-1], src_wavs.shape[-1], min_samples,
        )

    pred_trimmed = pred_wavs[..., :min_samples]
    src_trimmed = src_wavs[..., :min_samples].to(
        device=pred_trimmed.device, dtype=pred_trimmed.dtype,
    )

    crossfade_samples = int(crossfade_duration * sample_rate)
    result = pred_trimmed.clone()
    to_sample = round if preserve_source else int

    for b in range(B):
        start_sample = to_sample(repainting_starts[b] * sample_rate)
        end_sample = to_sample(repainting_ends[b] * sample_rate)
        start_sample = max(0, min(start_sample, min_samples))
        end_sample = max(start_sample, min(end_sample, min_samples))

        if start_sample == 0 and end_sample >= splice_samples:
            continue

        mask = _build_waveform_crossfade_mask(
            splice_samples, start_sample, end_sample, crossfade_samples,
            device=pred_trimmed.device,
            crossfade_inside=preserve_source,
        )
        mask = mask[:min_samples]
        m = mask.unsqueeze(0).expand_as(result[b])
        result[b] = m * pred_trimmed[b] + (1.0 - m) * src_trimmed[b]

    if preserve_source and src_wavs.shape[-1] > min_samples:
        result = torch.cat([result, src_wavs[..., min_samples:].to(result)], dim=-1)
    elif not preserve_source and pred_wavs.shape[-1] > min_samples:
        result = torch.cat(
            [result, pred_wavs[..., min_samples:]], dim=-1,
        )

    return result
