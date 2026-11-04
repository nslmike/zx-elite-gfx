import os
import unittest
from unittest.mock import patch

from elite_lab.model import COBRA_IDS
from elite_lab.viewer import Viewer, pygame


def det(a, b, c):
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


class CobraWingTest(unittest.TestCase):
    def test_original_wing_edge_in_both_views(self):
        with patch.dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy"):
            self.addCleanup(pygame.quit)
            viewer = Viewer()

            def press(key):
                viewer.event(pygame.event.Event(pygame.KEYDOWN, key=key))

            for model_id in COBRA_IDS:
                viewer.choose_model(model_id)
                press(pygame.K_f)
                press(pygame.K_SPACE)
                press(pygame.K_w)
                viewer.render()
                self.assertEqual((viewer.model_id, viewer.edge_index, viewer.face_test), (model_id, 20, "elite"))
                self.assertFalse(viewer.rotate_running)
                edge = viewer.model.edges[20]
                self.assertEqual((edge.vertex_a, edge.vertex_b, edge.faces), (1, 5, (3, 1)))
                flags = viewer.geometry.face_flags
                self.assertEqual((flags[3], flags[1]), (1, 0))
                trace = viewer.ship_lines[20]
                self.assertEqual((trace.start, trace.end, trace.clip_status, trace.count),
                                 ((156, 46), (80, 40), "inside", 76))

                # Луч к середине сгиба в координатах камеры сперва проходит внутри грани 5.
                points = viewer.geometry.details["transformed"]
                mid = tuple((x + y) / 2 for x, y in zip(points[1], points[5]))
                a, b, c = (points[i] for i in (2, 5, 9))
                # Представляем mid = u*A + v*B + w*C. Деление на сумму весов даёт
                # точку пересечения с плоскостью в барицентрических координатах.
                volume = det(a, b, c)
                weights = (det(mid, b, c) / volume, det(a, mid, c) / volume, det(a, b, mid) / volume)
                self.assertTrue(all(w > 0 for w in weights))
                self.assertLess(1 / sum(weights), 1)

                press(pygame.K_RETURN)
                self.assertEqual(viewer.line_trace.pixels, trace.pixels)
                press(pygame.K_1)
                press(pygame.K_w)
                viewer.render()
                self.assertEqual((viewer.yaw, viewer.pitch, viewer.roll, viewer.depth, viewer.edge_index),
                                 (60, -25, 0, 384, 20))
                flags = viewer.geometry.face_flags
                self.assertEqual((flags[3], flags[1]), (0, 0))
                self.assertEqual(viewer.geometry.edge_decisions[20].reason, "faces")
                self.assertNotIn(20, viewer.ship_lines)
                press(pygame.K_w)
                self.assertEqual(viewer.ship_lines[20].pixels, trace.pixels)
