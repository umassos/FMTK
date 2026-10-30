"""Tests for TaskGraph and the Pipeline task API."""
import unittest

import torch
import torch.nn as nn

from fmtk.pipeline import Pipeline
from fmtk.task_graph import TaskGraph


class Counted(nn.Module):
    """Linear layer that counts forward calls."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.lin(x)


class Backbone(Counted):
    """Backbone with the FMTK forward(x, mask) signature."""

    device = "cpu"

    def forward(self, x, mask=None):
        return super().forward(x)


def two_backbone_graph():
    g = TaskGraph()
    parts = {"bb_a": Counted(4, 6), "bb_b": Counted(3, 6), "comp": Counted(6, 6),
             "dec_a": Counted(6, 2), "dec_b": Counted(6, 5)}
    for name, module in parts.items():
        g.add_component(name, module)
    g.register_task("t1", ["bb_a", "comp", "dec_a"])
    g.register_task("t2", ["bb_b", "comp", "dec_b"])
    return g, parts


class TestTaskGraph(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_forward_matches_manual_calls(self):
        g, p = two_backbone_graph()
        x = torch.randn(3, 4)
        with torch.no_grad():
            expected = p["dec_a"](p["comp"](p["bb_a"](x)))
        torch.testing.assert_close(g.forward(x, "t1"), expected)

    def test_register_rejects_bad_paths(self):
        g, _ = two_backbone_graph()
        with self.assertRaisesRegex(ValueError, "unknown components"):
            g.register_task("bad", ["bb_a", "nope"])
        with self.assertRaisesRegex(ValueError, "empty path"):
            g.register_task("bad", [])
        with self.assertRaisesRegex(ValueError, "already exists"):
            g.add_component("comp", Counted(6, 6))

    def test_forward_tasks_matches_per_sample(self):
        g = TaskGraph()
        bb, dec_a, dec_b = Counted(4, 6), Counted(6, 2), Counted(6, 5)
        for name, m in (("bb", bb), ("dec_a", dec_a), ("dec_b", dec_b)):
            g.add_component(name, m)
        g.register_task("t1", ["bb", "dec_a"])
        g.register_task("t2", ["bb", "dec_b"])
        x = torch.randn(4, 4)
        tasks = ["t1", "t2", "t2", "t1"]
        outs = g.forward_tasks(x, tasks)
        for i, t in enumerate(tasks):
            torch.testing.assert_close(outs[i], g.forward(x[i:i + 1], t)[0])
        self.assertEqual([o.shape[0] for o in outs], [2, 5, 5, 2])

    def test_shared_components_run_once_per_wave(self):
        g, p = two_backbone_graph()
        x = torch.randn(3, 4)
        g.add_component("bb_b_in", lambda z: p["bb_b"](z[:, :3]))
        g.register_task("t2x", ["bb_b_in", "comp", "dec_b"])
        g.forward_tasks(x, ["t1", "t2x", "t1"])
        self.assertEqual(p["comp"].calls, 1)
        self.assertEqual((p["bb_a"].calls, p["dec_a"].calls, p["dec_b"].calls), (1, 1, 1))

    def test_on_output_fires_before_longer_paths_finish(self):
        g = TaskGraph()
        events = []
        g.add_component("bb", Counted(4, 6))
        g.add_component("head_a", Counted(6, 2))
        g.add_component("stitch", lambda z: (events.append("stitch"), z)[1])
        g.add_component("head_b", lambda z: (events.append("head_b"), z[:, :5])[1])
        g.register_task("short", ["bb", "head_a"])
        g.register_task("long", ["bb", "stitch", "head_b"])
        x = torch.randn(3, 4)
        outs = g.forward_tasks(x, ["short", "long", "short"], on_output=lambda i, y: events.append(i))
        self.assertLess(events.index(0), events.index("head_b"))
        self.assertLess(events.index(2), events.index("head_b"))
        self.assertEqual(events[-1], 1)
        self.assertEqual(sorted(e for e in events if isinstance(e, int)), [0, 1, 2])
        torch.testing.assert_close(outs[1], g.forward(x[1:2], "long")[0])

    def test_remove_task_drops_unused_components(self):
        g, _ = two_backbone_graph()
        with self.assertRaisesRegex(ValueError, "used by tasks"):
            g.remove_component("comp")
        g.remove_task("t1")
        self.assertNotIn("bb_a", g.components)
        self.assertNotIn("dec_a", g.components)
        self.assertIn("comp", g.components)
        g.remove_task("t2", drop_unused=False)
        self.assertIn("comp", g.components)


class TestPipelineTasks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.backbone = Backbone(4, 6)
        self.pipeline = Pipeline(self.backbone)
        self.head = Counted(6, 3)

    def test_task_path_through_backbone(self):
        stitch = Counted(6, 6)
        self.pipeline.add_component("stitch", stitch)
        self.pipeline.add_component("head", self.head)
        self.pipeline.register_task("t", ["backbone", "stitch", "head"])
        x = torch.randn(2, 4)
        with torch.no_grad():
            expected = self.head(stitch(self.backbone(x)))
        torch.testing.assert_close(self.pipeline.forward(x, task="t"), expected)

    def test_register_task_requires_backbone_then_component(self):
        self.pipeline.add_component("head", self.head)
        for path in ([], ["head"], ["backbone"], ["head", "backbone"], ["backbone", "backbone", "head"]):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "exactly one 'backbone'"):
                self.pipeline.register_task("t", path)
        self.assertEqual(self.pipeline.graph.tasks, {})

    def test_backbone_survives_removing_last_task(self):
        self.pipeline.add_component("head", self.head)
        self.pipeline.register_task("t", ["backbone", "head"])
        self.pipeline.remove_task("t")
        self.assertNotIn("head", self.pipeline.graph.components)
        self.pipeline.add_component("head", self.head)
        self.pipeline.register_task("t", ["backbone", "head"])
        self.assertEqual(self.pipeline.forward(torch.randn(2, 4), task="t").shape, (2, 3))

    def test_legacy_forward_unchanged(self):
        self.pipeline.add_decoder(self.head, load=True, train=True)
        x = torch.randn(2, 4)
        with torch.no_grad():
            expected = self.head(self.backbone(x))
        torch.testing.assert_close(self.pipeline.forward(x), expected)


if __name__ == "__main__":
    unittest.main()
