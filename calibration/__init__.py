"""Observer calibration (accessibility parameters, not diagnosis). SPEC 5.3."""
from .model import SelfReport, Prior, SimObserver, PARAMS
from .bayes import Calibrator
from .session import Session, CalibConfig
from .stimuli import pool, AXES, decode
