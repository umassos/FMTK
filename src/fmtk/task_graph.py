"""Named components and task paths through them."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

Groups = list[tuple[list[int], torch.Tensor]]


class TaskGraph:
    """Holds named components and runs registered task paths over them."""

    def __init__(self) -> None:
        self._components: dict[str, Any] = {}
        self._fns: dict[str, Callable] = {}
        self._tasks: dict[str, tuple[str, ...]] = {}
        self._task_dag: dict[str, tuple[Callable, ...]] = {}

    @property
    def components(self) -> dict[str, Any]:
        return dict(self._components)

    @property
    def tasks(self) -> dict[str, tuple[str, ...]]:
        return dict(self._tasks)

    def add_component(self, name: str, obj: Any) -> None:
        """Register an object with forward(x), or a plain callable, under name."""
        if name in self._components:
            raise ValueError(f"component {name!r} already exists")
        if isinstance(obj, torch.nn.Module):
            obj.eval()
        fn = obj.forward if hasattr(obj, "forward") else obj
        if not callable(fn):
            raise TypeError(f"component {name!r} has no forward and is not callable")
        self._components = {**self._components, name: obj}
        self._fns = {**self._fns, name: fn}

    def remove_component(self, name: str) -> None:
        """Remove a component that no task uses."""
        users = [t for t, path in self._tasks.items() if name in path]
        if users:
            raise ValueError(f"component {name!r} is used by tasks {users}")
        self._components = {k: v for k, v in self._components.items() if k != name}
        self._fns = {k: v for k, v in self._fns.items() if k != name}

    def register_task(self, task_id: str, path: Sequence[str]) -> None:
        """Validate a path of component names and add it to the task DAG."""
        path = tuple(path)
        if not path:
            raise ValueError(f"task {task_id!r}: empty path")
        missing = [n for n in path if n not in self._fns]
        if missing:
            raise ValueError(f"task {task_id!r}: unknown components {missing}")
        self._tasks = {**self._tasks, task_id: path}
        self._task_dag = {**self._task_dag, task_id: tuple(self._fns[n] for n in path)}

    def remove_task(self, task_id: str, drop_unused: bool = True) -> None:
        """Remove a task and, optionally, the components no remaining task uses."""
        path = self._tasks.get(task_id, ())
        self._tasks = {k: v for k, v in self._tasks.items() if k != task_id}
        self._task_dag = {k: v for k, v in self._task_dag.items() if k != task_id}
        if drop_unused:
            still_used = {n for p in self._tasks.values() for n in p}
            for name in set(path) - still_used:
                self.remove_component(name)

    def forward(self, x: Any, task: str) -> Any:
        """Run one task's path on a batch."""
        fns = self._task_dag[task]
        with torch.inference_mode():
            for fn in fns:
                x = fn(x)
        return x

    def forward_tasks(self, x: torch.Tensor, tasks: Sequence[str],
                      on_output: Callable[[int, torch.Tensor], None] | None = None) -> list[torch.Tensor]:
        """Run a mixed batch, one task per sample; shared components run once per wave.

        on_output(i, y) is called as soon as sample i finishes its path.
        """
        paths = [self._tasks[t] for t in tasks]
        pos = [0] * len(tasks)
        out: list[torch.Tensor | None] = [None] * len(tasks)
        pending: dict[str, Groups] = {}
        self._advance(list(range(len(tasks))), x, paths, pos, out, pending, on_output)
        with torch.inference_mode():
            while pending:
                for name in self._next_groups(pending):
                    ids, act = self._merge(pending.pop(name))
                    result = self._fns[name](act)
                    for i in ids:
                        pos[i] += 1
                    self._advance(ids, result, paths, pos, out, pending, on_output)
        return out

    def _next_groups(self, pending: dict[str, Groups]) -> list[str]:
        """Component names to run in the next wave."""
        return list(pending)

    @staticmethod
    def _merge(groups: Groups) -> tuple[list[int], torch.Tensor]:
        if len(groups) == 1:
            return groups[0]
        ids = [i for g, _ in groups for i in g]
        return ids, torch.cat([act for _, act in groups])

    @staticmethod
    def _advance(ids: list[int], act: torch.Tensor, paths: list[tuple[str, ...]], pos: list[int],
                 out: list, pending: dict[str, Groups], on_output: Callable | None) -> None:
        by_next: dict[str | None, list[int]] = {}
        for row, i in enumerate(ids):
            by_next.setdefault(paths[i][pos[i]] if pos[i] < len(paths[i]) else None, []).append(row)
        for name, rows in by_next.items():
            sub = act if len(rows) == len(ids) else act[torch.tensor(rows, device=act.device)]
            sub_ids = [ids[r] for r in rows]
            if name is None:
                for r, i in enumerate(sub_ids):
                    out[i] = sub[r]
                    if on_output is not None:
                        on_output(i, out[i])
            else:
                pending.setdefault(name, []).append((sub_ids, sub))
