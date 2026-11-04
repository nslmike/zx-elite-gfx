import os
import unittest
from unittest.mock import patch

from elite_lab.viewer import KRAIT_ID, Viewer, pygame


class LineRendererTest(unittest.TestCase):
    def test_krait_clipping_and_switch_to_pixel_inspector(self):
        with patch.dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy"):
            self.addCleanup(pygame.quit)
            viewer = Viewer()
            viewer.show_clipping_example(KRAIT_ID)
            viewer.zoom(3)
            pixel_frame = bytes(viewer.ship_frame.data)

            def press(key):
                viewer.event(pygame.event.Event(pygame.KEYDOWN, key=key))

            # Линейный кадр должен строиться без трассировки отдельных пикселей.
            with patch("elite_lab.viewer.trace_line", side_effect=AssertionError("pixel trace")):
                press(pygame.K_l)
                clipped = viewer.ship_lines[8]
                self.assertEqual((clipped.endpoints, clipped.count), (((242, 94), (253, 66)), 29))
                with patch.object(pygame.draw, "line", wraps=pygame.draw.line) as draw:
                    viewer.render()
                    self.assertEqual(draw.call_count, sum(s.count > 0 for s in viewer.ship_lines.values()))
                viewer.zoom(1)
                self.assertEqual(viewer.ship_lines[8].clip_status, "rejected")
                viewer.zoom(-1)

            press(pygame.K_RETURN)
            self.assertEqual(viewer.mode, "line")
            press(pygame.K_RIGHT)
            self.assertEqual((viewer.line_count, viewer.line_trace.count), (1, 29))
            press(pygame.K_1)
            self.assertEqual(viewer.renderer, "lines")
            press(pygame.K_l)
            self.assertEqual((viewer.depth, viewer.edge_index), (208, 8))
            self.assertEqual(bytes(viewer.ship_frame.data), pixel_frame)
