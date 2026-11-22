from .features import Tracker, Geometry, geometry, FEATURE_NAMES, IDX, CLASSES  # noqa: F401
from .rules import rule_score  # noqa: F401
from .task_observer import TaskParams, labels as task_labels, dprime, reaction_time  # noqa: F401
from .learned import NumpyMLP, train as train_mlp, MODEL_PATH  # noqa: F401


def load_scorer():
    """Learned MLP if bench/importance_report.py found it beats the rules on held-out scenes (mlp.npz exists), else rules."""
    return NumpyMLP.load() if MODEL_PATH.exists() else rule_score
