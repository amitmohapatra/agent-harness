"""Evaluation: the online judge, the offline experiment, and the gate between them."""

from trellis.harness.evaluation.answers import answer_text
from trellis.harness.evaluation.budget import Admission, JudgeBudget
from trellis.harness.evaluation.datasets import Dataset, DatasetBuilder, DatasetItem
from trellis.harness.evaluation.events import (
    CollectingEvaluationSink,
    CompositeEvaluationSink,
    LifecycleDispatcher,
    LoggingEvaluationSink,
    NoOpEvaluationProvider,
    build_eval_event,
)
from trellis.harness.evaluation.experiments import (
    ExperimentResult,
    ExperimentRunner,
    ItemResult,
)
from trellis.harness.evaluation.gate import (
    GateFinding,
    GateReport,
    GateThresholds,
    RegressionGate,
)
from trellis.harness.evaluation.grounding import GroundedDecision
from trellis.harness.evaluation.judge import GroundedJudge, GroundingVerifier, ScopedJudge
from trellis.harness.evaluation.rubric import BUILTIN_RUBRIC, RubricPrompt

__all__ = [
    "BUILTIN_RUBRIC",
    "Admission",
    "CollectingEvaluationSink",
    "CompositeEvaluationSink",
    "Dataset",
    "DatasetBuilder",
    "DatasetItem",
    "ExperimentResult",
    "ExperimentRunner",
    "GateFinding",
    "GateReport",
    "GateThresholds",
    "GroundedDecision",
    "GroundedJudge",
    "GroundingVerifier",
    "ItemResult",
    "JudgeBudget",
    "LifecycleDispatcher",
    "LoggingEvaluationSink",
    "NoOpEvaluationProvider",
    "RegressionGate",
    "RubricPrompt",
    "ScopedJudge",
    "answer_text",
    "build_eval_event",
]
