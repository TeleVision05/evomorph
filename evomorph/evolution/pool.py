"""Batch evaluation over a process pool (at most ``MAX_WORKERS`` processes).

Results come back in input order (``Executor.map``), and every random draw of
the algorithms happens in the parent process, so a run is identical with 1 or 6
workers and whatever order the workers finish in. With one worker the batch
runs in-process (no pool start-up, easier debugging).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from types import TracebackType

from evomorph.evolution.evalspec import Evaluator
from evomorph.morphology.genome import Genome
from evomorph.simulation.evaluator import EvalResult

# House rule: the owner works on this laptop during runs, so no pool ever uses
# more than 6 processes, whatever a config or command line asks for.
MAX_WORKERS = 6


def clamp_workers(requested: int) -> int:
    """``requested`` limited to [1, MAX_WORKERS]."""
    return max(1, min(int(requested), MAX_WORKERS))


class BatchRunner:
    """Context manager that maps an evaluator over batches of genomes."""

    def __init__(self, evaluator: Evaluator, workers: int = 1) -> None:
        self.evaluator = evaluator
        self.workers = clamp_workers(workers)
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> BatchRunner:
        if self.workers > 1:
            self._pool = ProcessPoolExecutor(max_workers=self.workers)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def map(self, genomes: Sequence[Genome]) -> list[EvalResult]:
        """Evaluate ``genomes``; results are in the same order."""
        if not genomes:
            return []
        if self._pool is None:
            return [self.evaluator(g) for g in genomes]
        # chunksize 1: evaluations vary 10x in cost (falls end early), so small
        # chunks balance the load; pickling a genome is negligible next to 0.2 s.
        return list(self._pool.map(self.evaluator, genomes, chunksize=1))
