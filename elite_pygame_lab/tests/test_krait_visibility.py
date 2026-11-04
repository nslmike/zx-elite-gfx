import os
import unittest
from unittest.mock import patch

from elite_lab.controls import pose_from_angles
from elite_lab.model import KRAIT_ID, MODELS
from elite_lab.renderer import render_geometry
from elite_lab.viewer import Viewer, pygame


class KraitVisibilityTest(unittest.TestCase):
    def test_underside_views_and_historical_modes(self):
        with patch.dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy"):
            self.addCleanup(pygame.quit)
            viewer = Viewer()
            viewer.choose_model(KRAIT_ID)

            def press(key):
                viewer.event(pygame.event.Event(pygame.KEYDOWN, key=key))

            for old_count, start, end in ((14, (37, 61), (8, 47)), (11, (52, 69), (-13, 47))):
                press(pygame.K_u)
                pose = (viewer.yaw, viewer.pitch, viewer.roll, viewer.depth, viewer.edge_index)
                for mode, count in (("hull", 7), ("elite", old_count), ("vertex", old_count)):
                    self.assertEqual(viewer.face_test, mode)
                    for _ in range(2):
                        viewer.render()
                        drawn = {i for i, line in viewer.ship_lines.items() if line.count}
                        self.assertEqual(len(drawn), count)
                        if mode == "hull":
                            self.assertEqual(viewer.geometry.face_flags[:6], [0, 1, 1, 0, 0, 0])
                            self.assertEqual(drawn, {1, 2, 3, 5, 6, 8, 9})
                        edge = viewer.geometry.edge_decisions[8]
                        self.assertEqual(((edge.start.x, edge.start.y), (edge.end.x, edge.end.y)), (start, end))
                        press(pygame.K_l)
                    press(pygame.K_f)
                    self.assertEqual((viewer.yaw, viewer.pitch, viewer.roll, viewer.depth, viewer.edge_index), pose)

    def test_top_panels_return_when_seen_from_above(self):
        model = MODELS[KRAIT_ID]
        for roll in (0, 45, 180):
            result = render_geometry(model, pose_from_angles(0, -90, roll, 256))
            self.assertEqual(result.face_flags[:6], [1, 0, 0, 1, 0, 0])
            self.assertTrue(all(result.edge_decisions[i].draw for i in range(11, 15)))
