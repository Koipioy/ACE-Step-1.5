"""Execution helper for ``generate_music`` service invocation with progress tracking."""

from typing import Any, Dict, List, Optional, Sequence

from ..generation_watchdog import StepCallback, run_generation_worker


class GenerateMusicExecuteMixin:
    """Run service generation under diffusion progress estimation lifecycle."""

    def _run_generate_music_service_with_progress(
        self,
        progress: Any,
        actual_batch_size: int,
        audio_duration: Optional[float],
        inference_steps: int,
        timesteps: Optional[Sequence[float]],
        service_inputs: Dict[str, Any],
        refer_audios: Optional[List[Any]],
        guidance_scale: float,
        actual_seed_list: Optional[List[int]],
        audio_cover_strength: float,
        cover_noise_strength: float,
        use_adg: bool,
        cfg_interval_start: float,
        cfg_interval_end: float,
        shift: float,
        infer_method: str,
        sampler_mode: str = "euler",
        velocity_norm_threshold: float = 0.0,
        velocity_ema_factor: float = 0.0,
        dcw_enabled: bool = True,
        dcw_mode: str = "double",
        dcw_scaler: float = 0.05,
        dcw_high_scaler: float = 0.02,
        dcw_wavelet: str = "haar",
        repaint_crossfade_frames: int = 10,
        repaint_injection_ratio: float = 0.5,
        source_repaint_latents: Any = None,
        task_type: str = "",
        actual_retake_seed_list: Optional[List[int]] = None,
        retake_variance: float = 0.0,
        flow_edit_morph: bool = False,
        flow_edit_source_caption: str = "",
        flow_edit_source_lyrics: str = "",
        flow_edit_n_min: float = 0.0,
        flow_edit_n_max: float = 1.0,
        flow_edit_n_avg: int = 1,
    ) -> Dict[str, Any]:
        """Invoke ``service_generate`` with progress reporting and timeout protection.

        Use completed steps when the sampler reports them. Other backends use
        estimated progress so the interface continues to update.
        """
        infer_steps_for_progress = len(timesteps) if timesteps else inference_steps
        progress_desc = f"Generating music (batch size: {actual_batch_size})..."
        progress(0.52, desc=progress_desc)
        stop_event = None
        progress_thread = None
        reports_steps = (
            getattr(getattr(self, "model", None), "reports_diffusion_steps", False) is True
            and not (getattr(self, "use_mlx_dit", False) and self.mlx_decoder is not None)
            and not flow_edit_morph
        )

        def generate(on_diffusion_step: Optional[StepCallback]) -> Dict[str, Any]:
            """Forward prepared inputs and the optional sampler callback to the worker."""
            return self.service_generate(
                captions=service_inputs["captions_batch"],
                global_captions=service_inputs.get("global_captions_batch"),
                lyrics=service_inputs["lyrics_batch"],
                metas=service_inputs["metas_batch"],
                vocal_languages=service_inputs["vocal_languages_batch"],
                refer_audios=refer_audios,
                target_wavs=service_inputs["target_wavs_tensor"],
                infer_steps=inference_steps,
                guidance_scale=guidance_scale,
                seed=actual_seed_list,
                repainting_start=service_inputs["repainting_start_batch"],
                repainting_end=service_inputs["repainting_end_batch"],
                instructions=service_inputs["instructions_batch"],
                audio_cover_strength=audio_cover_strength,
                cover_noise_strength=cover_noise_strength,
                use_adg=use_adg,
                cfg_interval_start=cfg_interval_start,
                cfg_interval_end=cfg_interval_end,
                shift=shift,
                infer_method=infer_method,
                sampler_mode=sampler_mode,
                velocity_norm_threshold=velocity_norm_threshold,
                velocity_ema_factor=velocity_ema_factor,
                dcw_enabled=dcw_enabled,
                dcw_mode=dcw_mode,
                dcw_scaler=dcw_scaler,
                dcw_high_scaler=dcw_high_scaler,
                dcw_wavelet=dcw_wavelet,
                audio_code_hints=service_inputs["audio_code_hints_batch"],
                return_intermediate=service_inputs["should_return_intermediate"],
                timesteps=timesteps,
                chunk_mask_modes=service_inputs.get("chunk_mask_modes_batch"),
                repaint_crossfade_frames=repaint_crossfade_frames,
                repaint_injection_ratio=repaint_injection_ratio,
                source_repaint_latents=source_repaint_latents,
                task_type=task_type,
                retake_seed=actual_retake_seed_list,
                retake_variance=retake_variance,
                flow_edit_morph=flow_edit_morph,
                flow_edit_source_caption=flow_edit_source_caption,
                flow_edit_source_lyrics=flow_edit_source_lyrics,
                flow_edit_n_min=flow_edit_n_min,
                flow_edit_n_max=flow_edit_n_max,
                flow_edit_n_avg=flow_edit_n_avg,
                on_diffusion_step=on_diffusion_step,
            )

        def report_step(completed: int, total: int) -> None:
            """Show the latest real diffusion step in the generation UI."""
            progress(
                0.52 + (0.79 - 0.52) * completed / total,
                desc=f"Diffusion step {completed}/{total} (batch size: {actual_batch_size})",
            )

        try:
            if not reports_steps:
                stop_event, progress_thread = self._start_diffusion_progress_estimator(
                    progress=progress,
                    start=0.52,
                    end=0.79,
                    infer_steps=infer_steps_for_progress,
                    batch_size=actual_batch_size,
                    duration_sec=audio_duration if audio_duration and audio_duration > 0 else None,
                    desc=progress_desc,
                )
            outputs = run_generation_worker(
                self, generate, report_step if reports_steps else None,
                f"batch={actual_batch_size}, steps={inference_steps}, duration={audio_duration}s",
            )
        finally:
            if stop_event is not None:
                stop_event.set()
            if progress_thread is not None:
                progress_thread.join(timeout=1.0)

        return {"outputs": outputs, "infer_steps_for_progress": infer_steps_for_progress}
