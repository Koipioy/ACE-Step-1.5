"""Check the layout and visibility boundaries used by the Repaint workflow."""

import unittest
from unittest.mock import MagicMock

import gradio as gr

from .generation_tab_section import create_generation_tab_section


class RepaintLayoutTests(unittest.TestCase):
    """Keep Repaint inputs in workflow order and Flow-Edit in its own column."""

    def setUp(self) -> None:
        """Build the tab with an empty model catalog and no loaded weights."""
        handler = MagicMock()
        handler.get_available_acestep_v15_models.return_value = []
        with gr.Blocks():
            self.controls = create_generation_tab_section(handler, None)

    def test_source_range_lyrics_and_action_share_a_logical_order(self) -> None:
        """Put the action immediately after the prompt and before optional settings."""
        siblings = self.controls["repainting_group"].parent.children
        fields = (
            "src_audio_row", "repainting_group", "custom_mode_group",
            "generate_btn_row", "optional_params_accordion",
        )
        positions = [siblings.index(self.controls[name]) for name in fields]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(positions[3], positions[2] + 1)

    def test_flow_edit_visibility_covers_the_source_lyrics_panel(self) -> None:
        """A mode change hides the whole Flow-Edit panel and leaves Retake available."""
        column = self.controls["flow_edit_column"]
        self.assertIs(self.controls["morph_panel"].parent, column)
        self.assertIsNot(self.controls["retake_panel"].parent, column)
