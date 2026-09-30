"""Evaluation: the sampled online judge, offline datasets and experiments, and the
regression gate CI runs (``python -m trellis.eval gate``)."""

from trellis.eval.budget import JudgeBudget
from trellis.eval.datasets import Dataset, DatasetBuilder, DatasetItem
from trellis.eval.experiments import ExperimentResult, ExperimentRunner, ItemResult
from trellis.eval.gate import GateReport, GateThresholds, RegressionGate
from trellis.eval.judge import GroundedJudge

__all__ = [
    "Dataset",
    "DatasetBuilder",
    "DatasetItem",
    "ExperimentResult",
    "ExperimentRunner",
    "GateReport",
    "GateThresholds",
    "GroundedJudge",
    "ItemResult",
    "JudgeBudget",
    "RegressionGate",
]
